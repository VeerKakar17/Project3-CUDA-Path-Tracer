#include "bvh.h"
#include "pathtrace.h"

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "interactions.h"
#include "intersections.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#define ERRORCHECK 1
static constexpr bool SORT_BY_MATERIAL = false;

#define FILENAME                                                               \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess == err) {
    return;
  }

  fprintf(stderr, "CUDA error");
  if (file) {
    fprintf(stderr, " (%s:%d)", file, line);
  }
  fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
  getchar();
#endif // _WIN32
  exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__ thrust::default_random_engine
makeSeededRandomEngine(int iter, int index, int depth) {
  int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
  return thrust::default_random_engine(h);
}

struct IsActiveMatIdx {
  __host__ __device__ bool operator()(uint8_t matidx) const {
    return matidx != 255;
  }
};

struct IsActivePath {
  __host__ __device__ bool operator()(const PathSegment &path) const {
    return path.remainingBounces > 0;
  }
};

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution, int iter,
                               glm::vec3 *image) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);
    glm::vec3 pix = image[index];

    glm::ivec3 color;
    color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
    color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
    color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

    // Each thread writes one pixel location in the texture (textel)
    pbo[index].w = 0;
    pbo[index].x = color.x;
    pbo[index].y = color.y;
    pbo[index].z = color.z;
  }
}

static Scene *hst_scene = NULL;
static GuiDataContainer *guiData = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms = NULL;
static Triangle *dev_triangles = NULL;
static Material *dev_materials = NULL;
static BVHNode *dev_bvh = NULL;
static PathSegment *dev_paths = NULL;
static PathSegment *dev_paths_tmp = NULL;
static PathSegment *dev_paths_tmp2 = NULL;
static ShadeableIntersection *dev_intersections = NULL;
static ShadeableIntersection *dev_intersections_tmp = NULL;
static int *dev_firstThreadIdx = NULL;
static uint8_t *dev_segment_matidx = NULL;
static uint8_t *dev_intersection_matidx = NULL;
static thrust::device_ptr<uint8_t> dev_thrust_segment_matidx = NULL;
static thrust::device_ptr<uint8_t> dev_thrust_intersection_matidx = NULL;
static thrust::device_ptr<PathSegment> dev_thrust_paths = NULL;
static thrust::device_ptr<PathSegment> dev_thrust_paths_tmp = NULL;
static thrust::device_ptr<PathSegment> dev_thrust_paths_tmp2 = NULL;
static thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections =
    NULL;
static thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections_tmp =
    NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer *imGuiData) { guiData = imGuiData; }

void pathtraceInit(Scene *scene) {
  hst_scene = scene;

  const Camera &cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
  cudaMalloc(&dev_paths_tmp, pixelcount * sizeof(PathSegment));
  cudaMalloc(&dev_paths_tmp2, pixelcount * sizeof(PathSegment));

  cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
  cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(),
             scene->materials.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_bvh, scene->bvh.nodes.size() * sizeof(BVHNode));
  cudaMemcpy(dev_bvh, scene->bvh.nodes.data(), scene->bvh.nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
  cudaMalloc(&dev_intersections_tmp, pixelcount * sizeof(ShadeableIntersection));
  cudaMemset(dev_intersections_tmp, 0, pixelcount * sizeof(ShadeableIntersection));


  cudaMalloc(&dev_firstThreadIdx, sizeof(int) * 1);
  cudaMalloc(&dev_segment_matidx, sizeof(uint8_t) * pixelcount);
  cudaMalloc(&dev_intersection_matidx, sizeof(uint8_t) * pixelcount);

  dev_thrust_segment_matidx = thrust::device_pointer_cast(dev_segment_matidx);
  dev_thrust_intersection_matidx =
      thrust::device_pointer_cast(dev_intersection_matidx);
  dev_thrust_paths = thrust::device_pointer_cast(dev_paths);
  dev_thrust_paths_tmp = thrust::device_pointer_cast(dev_paths_tmp);
  dev_thrust_paths_tmp2 = thrust::device_pointer_cast(dev_paths_tmp2);
  dev_thrust_intersections = thrust::device_pointer_cast(dev_intersections);
  dev_thrust_intersections_tmp =
      thrust::device_pointer_cast(dev_intersections_tmp);

  // TODO: initialize any extra device memeory you need

  checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
  cudaFree(dev_image); // no-op if dev_image is null
  cudaFree(dev_paths);
  cudaFree(dev_paths_tmp);
  cudaFree(dev_paths_tmp2);
  cudaFree(dev_geoms);
  cudaFree(dev_triangles);
  cudaFree(dev_bvh);
  cudaFree(dev_materials);
  cudaFree(dev_intersections);
  cudaFree(dev_intersections_tmp);
  cudaFree(dev_firstThreadIdx);
  cudaFree(dev_segment_matidx);
  cudaFree(dev_intersection_matidx);
  // TODO: clean up any extra device memory you created

  checkCUDAError("pathtraceFree");
}

/**
 * Generate PathSegments with rays from the camera through the screen into the
 * scene, which is the first bounce of rays.
 *
 * Antialiasing - add rays for sub-pixel sampling
 * motion blur - jitter rays "in time"
 * lens effect - jitter ray origin positions based on a lens
 */
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth,
                                      PathSegment *pathSegments) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < cam.resolution.x && y < cam.resolution.y) {
    int index = x + (y * cam.resolution.x);
    PathSegment &segment = pathSegments[index];

    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

    // TODO: implement antialiasing by jittering the ray

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
    thrust::uniform_real_distribution<float> u05(-0.5, 0.5);

    segment.ray.direction = cam.view -
                       cam.right * cam.pixelLength.x *
                           ((float)x - (float)cam.resolution.x * 0.5f) -
                       cam.up * cam.pixelLength.y *
                           ((float)y - (float)cam.resolution.y * 0.5f);

    segment.ray.direction.x += cam.pixelLength.x * u05(rng);
    segment.ray.direction.y += cam.pixelLength.y * u05(rng);
    segment.ray.direction = glm::normalize(segment.ray.direction);

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
  }
}

__device__ float IntersectAABB(const Ray &ray, const BVHNode &node,
                               float maxT) {
  float tmin = 0.0f;
  float tmax = maxT;

  for (int axis = 0; axis < 3; ++axis) {
    float origin = ray.origin[axis];
    float direction = ray.direction[axis];
    float minBound = node.aabbMin[axis];
    float maxBound = node.aabbMax[axis];

    if (fabsf(direction) < 0.0000001f) {
      if (origin < minBound || origin > maxBound) {
        return FLT_MAX;
      }
      continue;
    }

    float invDir = 1.0f / direction;
    float t1 = (minBound - origin) * invDir;
    float t2 = (maxBound - origin) * invDir;
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));
  }

  if (tmax >= tmin && tmin < maxT) {
    return tmin;
  }
  return FLT_MAX;
}

__device__ bool BVHIntersect(const Ray &ray, Triangle *triangles,
                             int triangles_size, BVHNode *bvh, float &tMin,
                             glm::vec3 &intersectPoint, glm::vec3 &normal,
                             int &materialId) {
  bool hit = false;
  int nodeIdx = 0;
  constexpr int STACK_SIZE = 128;
  int stack[STACK_SIZE];
  int stackPtr = 0;

  while (true) {
    BVHNode &node = bvh[nodeIdx];

    if (node.isLeaf()) {
      for (uint32_t i = 0; i < node.triCount; ++i) {
        int triangleIdx = (int)(node.firstTriIdx + i);
        if (triangleIdx >= triangles_size) {
          continue;
        }

        glm::vec3 tmpIntersect;
        glm::vec3 tmpNormal;
        bool outside = true;
        float t = triangleIntersectionTest(triangles[triangleIdx], ray,
                                           tmpIntersect, tmpNormal, outside);
        if (t > 0.0f && t < tMin) {
          hit = true;
          tMin = t;
          intersectPoint = tmpIntersect;
          normal = tmpNormal;
          materialId = triangles[triangleIdx].materialid;
        }
      }

      if (stackPtr == 0) {
        break;
      }
      nodeIdx = stack[--stackPtr];
      continue;
    }

    int child1 = (int)node.leftNode;
    int child2 = child1 + 1;
    float dist1 = IntersectAABB(ray, bvh[child1], tMin);
    float dist2 = IntersectAABB(ray, bvh[child2], tMin);

    if (dist1 > dist2) {
      float tempDist = dist1;
      dist1 = dist2;
      dist2 = tempDist;

      int tempChild = child1;
      child1 = child2;
      child2 = tempChild;
    }

    if (dist1 == FLT_MAX) {
      if (stackPtr == 0) {
        break;
      }
      nodeIdx = stack[--stackPtr];
    } else {
      nodeIdx = child1;
      if (dist2 != FLT_MAX && stackPtr < STACK_SIZE) {
        stack[stackPtr++] = child2;
      }
    }
  }

  return hit;
}

__global__ void computeIntersections(int depth, int num_paths,
                                     PathSegment *pathSegments, Geom *geoms,
                                     int geoms_size, Triangle *triangles,
                                     int triangles_size,
                                     ShadeableIntersection *intersections,
                                     PathSegment *orderedPathSegments, BVHNode *bvh) {
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index < num_paths) {
    PathSegment &pathSegment = pathSegments[path_index];

    if (pathSegment.remainingBounces > 0) {
      float t;
      glm::vec3 intersect_point;
      glm::vec3 normal;
      float t_min = FLT_MAX;
      int hit_geom_index = -1;
      int hit_material_id = -1;
      bool outside = true;

      glm::vec3 tmp_intersect;
      glm::vec3 tmp_normal;

      // naive parse through global geoms

      for (int i = 0; i < geoms_size; i++) {
        Geom &geom = geoms[i];

        if (geom.type == CUBE) {
          t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect,
                                  tmp_normal, outside);
        } else if (geom.type == SPHERE) {
          t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect,
                                     tmp_normal, outside);
        }
        // TODO: add more intersection tests here... triangle? metaball? CSG?

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t) {
          t_min = t;
          hit_geom_index = i;
          hit_material_id = geom.materialid;
          intersect_point = tmp_intersect;
          normal = tmp_normal;
        }
      }

      if (BVHIntersect(pathSegment.ray, triangles, triangles_size, bvh, t_min,
                       tmp_intersect, tmp_normal, hit_material_id)) {
        hit_geom_index = -1;
        intersect_point = tmp_intersect;
        normal = tmp_normal;
      }

      if (hit_material_id == -1) {
        intersections[path_index].t = -1.0f;
        intersections[path_index].geomId = -1;
        pathSegment.color = glm::vec3(0);
        pathSegment.remainingBounces = 0;
        if (orderedPathSegments != NULL) {
          orderedPathSegments[pathSegment.pixelIndex] = pathSegment;
        }
      } else {
        // The ray hits something
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = hit_material_id;
        intersections[path_index].geomId = hit_geom_index;
        intersections[path_index].surfaceNormal = normal;
      }
    }
  }
}

__global__ void computeRayColors(int iter, int num_paths,
                                 ShadeableIntersection *shadeableIntersections,
                                 PathSegment *pathSegments,
                                 Geom *geoms,
                                 Material *materials,
                                 PathSegment *orderedPathSegments) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths) {
      ShadeableIntersection intersection = shadeableIntersections[idx];
      PathSegment *segment = &pathSegments[idx];
      if (segment->remainingBounces > 0) {
        thrust::default_random_engine rng = makeSeededRandomEngine(
            iter, segment->pixelIndex, segment->remainingBounces);
        Material material = materials[intersection.materialId];
        if (intersection.t > 0.0f) {

            glm::vec3 old_dir = -segment->ray.direction;
            glm::vec3 intersect_point = getPointOnRay(segment->ray, intersection.t);

            thrust::uniform_real_distribution<float> u01(0, 1);
            if (material.alpha < 1.0f && u01(rng) > material.alpha) {
                glm::vec3 rayDir = glm::normalize(segment->ray.direction);
                segment->ray.origin = intersect_point + 0.0002f * rayDir;
                if (intersection.geomId >= 0) {
                  Geom geom = geoms[intersection.geomId];
                  Ray exitRay;
                  exitRay.origin = segment->ray.origin;
                  exitRay.direction = segment->ray.direction;

                  float exitT = -1.0f;
                  glm::vec3 exitPoint;
                  glm::vec3 exitNormal;
                  bool exitOutside = false;
                  if (geom.type == CUBE) {
                    exitT = boxIntersectionTest(geom, exitRay, exitPoint,
                                                exitNormal, exitOutside);
                  } else if (geom.type == SPHERE) {
                    exitT = sphereIntersectionTest(geom, exitRay, exitPoint,
                                                   exitNormal, exitOutside);
                  }

                  if (exitT > 0.0f) {
                    segment->ray.origin = exitPoint + 0.0002f * rayDir;
                  }
                }
                segment->remainingBounces--;
                if (orderedPathSegments != NULL) {
                  orderedPathSegments[segment->pixelIndex] = *segment;
                }
                return;
            }

            scatterRay(*segment, intersect_point, intersection.surfaceNormal, material,
                    rng);
            if (material.is_emissive) {
                segment->color *= (material.color * material.emissive_factor);
                segment->remainingBounces = 0;
            } else {
                float cos_angle = glm::dot(intersection.surfaceNormal, old_dir);
                segment->color *= material.color;
            }



            segment->remainingBounces--;
            if (orderedPathSegments != NULL) {
              orderedPathSegments[segment->pixelIndex] = *segment;
            }
        }
      }
  }
}

__global__ void mapToMatIdx(int num_paths, PathSegment *dev_segments, ShadeableIntersection *dev_intersections, uint8_t *dev_segment_matidx, uint8_t *dev_intersection_matidx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        uint8_t matidx;
        if (dev_segments[idx].remainingBounces <= 0) {
            matidx = 255;
        } else {
            matidx = dev_intersections[idx].materialId;
        }
        dev_segment_matidx[idx] = matidx;
        dev_intersection_matidx[idx] = matidx;
    }
}

__global__ void determineFirstThread(int num_paths, uint8_t *dev_matidx, int *dev_firstThreadIdx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        if (dev_matidx[idx] == 255) {
            atomicMin(dev_firstThreadIdx, idx);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *image,
                            PathSegment *iterationPaths) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths) {
    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color;
  }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
  const int traceDepth = hst_scene->state.traceDepth;
  const Camera &cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  // 2D block for generating ray from camera
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // 1D block for path tracing
  const int blockSize1d = 128;

  ///////////////////////////////////////////////////////////////////////////

  // Recap:
  // * Initialize array of path rays (using rays that come out of the camera)
  //   * You can pass the Camera object to that kernel.
  //   * Each path ray must carry at minimum a (ray, color) pair,
  //   * where color starts as the multiplicative identity, white = (1, 1, 1).
  //   * This has already been done for you.
  // * For each depth:
  //   * Compute an intersection in the scene for each path ray.
  //     A very naive version of this has been implemented for you, but feel
  //     free to add more primitives and/or a better algorithm.
  //     Currently, intersection distance is recorded as a parametric distance,
  //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
  //     * Color is attenuated (multiplied) by reflections off of any object
  //   * TODO: Stream compact away all of the terminated paths.
  //     You may use either your implementation or `thrust::remove_if` or its
  //     cousins.
  //     * Note that you can't really use a 2D kernel launch any more - switch
  //       to 1D.
  //   * TODO: Shade the rays that intersected something or didn't bottom out.
  //     That is, color the ray by performing a color computation according
  //     to the shader, then generate a new ray to continue the ray path.
  //     We recommend just updating the ray's PathSegment in place.
  //     Note that this step may come before or after stream compaction,
  //     since some shaders you write may also cause a path to terminate.
  // * Finally, add this iteration's results to the image. This has been done
  //   for you.

  // TODO: perform one iteration of path tracing

  generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth,
                                                          dev_paths);
  checkCUDAError("generate camera ray");

  int depth = 0;
  PathSegment *dev_path_end = dev_paths + pixelcount;
  int num_paths = dev_path_end - dev_paths;

  // --- PathSegment Tracing Stage ---
  // Shoot ray into scene, bounce between objects, push shading chunks

  int iteration_num = 0;
  int n = num_paths;
  PathSegment *dev_active_paths = dev_paths;
  PathSegment *dev_active_paths_tmp = dev_paths_tmp;
  thrust::device_ptr<PathSegment> dev_thrust_active_paths = dev_thrust_paths;
  thrust::device_ptr<PathSegment> dev_thrust_active_paths_tmp =
      dev_thrust_paths_tmp;

  if (!SORT_BY_MATERIAL) {
    thrust::copy(dev_thrust_paths, dev_thrust_paths + num_paths,
                 dev_thrust_paths_tmp);
    dev_active_paths = dev_paths_tmp;
    dev_active_paths_tmp = dev_paths_tmp2;
    dev_thrust_active_paths = dev_thrust_paths_tmp;
    dev_thrust_active_paths_tmp = dev_thrust_paths_tmp2;
  }

  ShadeableIntersection *dev_active_intersections = dev_intersections;
  ShadeableIntersection *dev_active_intersections_tmp = dev_intersections_tmp;
  thrust::device_ptr<ShadeableIntersection> dev_thrust_active_intersections =
      dev_thrust_intersections;
  thrust::device_ptr<ShadeableIntersection> dev_thrust_active_intersections_tmp =
      dev_thrust_intersections_tmp;

  while (iteration_num <= traceDepth) {
    // clean shading chunks
    cudaMemset(dev_active_intersections, 0,
               pixelcount * sizeof(ShadeableIntersection));

    // tracing
    dim3 numblocksPathSegmentTracing =
        (n + blockSize1d - 1) / blockSize1d;
    computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
        depth, n, dev_active_paths, dev_geoms, hst_scene->geoms.size(),
        dev_triangles, hst_scene->triangles.size(),
        dev_active_intersections, SORT_BY_MATERIAL ? NULL : dev_paths, dev_bvh);
    checkCUDAError("trace one bounce");
    // cudaDeviceSynchronize();
    depth++;

    if (SORT_BY_MATERIAL) {
      mapToMatIdx<<<numblocksPathSegmentTracing, blockSize1d>>>(
          n, dev_active_paths, dev_active_intersections, dev_segment_matidx,
          dev_intersection_matidx);
      thrust::stable_sort_by_key(dev_thrust_segment_matidx,
                                 dev_thrust_segment_matidx + n,
                                 dev_thrust_active_paths);
      thrust::stable_sort_by_key(dev_thrust_intersection_matidx,
                                 dev_thrust_intersection_matidx + n,
                                 dev_thrust_active_intersections);
      int firstThreadIdx = n;
      cudaMemcpy(dev_firstThreadIdx, &firstThreadIdx, sizeof(int),
                 cudaMemcpyHostToDevice);
      determineFirstThread<<<numblocksPathSegmentTracing, blockSize1d>>>(
          n, dev_segment_matidx, dev_firstThreadIdx);
      cudaMemcpy(&firstThreadIdx, dev_firstThreadIdx, sizeof(int),
                 cudaMemcpyDeviceToHost);
      n = firstThreadIdx;
      if (n == 0) {
          break;
      }
    } else {
      auto activePathsEnd =
          thrust::copy_if(dev_thrust_active_paths, dev_thrust_active_paths + n,
                          dev_thrust_active_paths, dev_thrust_active_paths_tmp,
                          IsActivePath());
      thrust::copy_if(dev_thrust_active_intersections,
                      dev_thrust_active_intersections + n,
                      dev_thrust_active_paths,
                      dev_thrust_active_intersections_tmp,
                      IsActivePath());

      n = activePathsEnd - dev_thrust_active_paths_tmp;

      PathSegment *tmp_paths = dev_active_paths;
      dev_active_paths = dev_active_paths_tmp;
      dev_active_paths_tmp = tmp_paths;
      
      ShadeableIntersection *tmp_intersections = dev_active_intersections;
      dev_active_intersections = dev_active_intersections_tmp;
      dev_active_intersections_tmp = tmp_intersections;
      
      dev_thrust_active_paths = thrust::device_pointer_cast(dev_active_paths);
      dev_thrust_active_paths_tmp =
          thrust::device_pointer_cast(dev_active_paths_tmp);
      dev_thrust_active_intersections =
          thrust::device_pointer_cast(dev_active_intersections);
      dev_thrust_active_intersections_tmp =
          thrust::device_pointer_cast(dev_active_intersections_tmp);
      
      if (n == 0) {
        break;
      }
    }

    numblocksPathSegmentTracing =
        (n + blockSize1d - 1) / blockSize1d;

    computeRayColors<<<numblocksPathSegmentTracing, blockSize1d>>>(
        iter, n, dev_active_intersections, dev_active_paths, dev_geoms,
        dev_materials, SORT_BY_MATERIAL ? NULL : dev_paths);

    if (guiData != NULL) {
      guiData->TracedDepth = depth;
    }

    iteration_num++;
  }

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
  finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image,
                                                dev_paths);

  ///////////////////////////////////////////////////////////////////////////

  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter,
                                                   dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
             pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathtrace");
}
