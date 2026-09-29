#include "interactions.h"

#include "intersections.h"
#include "microfacet.h"
#include "nee.h"
#include "utilities.h"

#include <thrust/random.h>

#define REMAP_ROUGHNESS 0
#define SHADOW_RAY_EPSILON 0.0002f

__device__ float power_heuristic(float pdfA, float pdfB) {
    float pdfA2 = pdfA * pdfA;
    float pdfB2 = pdfB * pdfB;
    float denom = pdfA2 + pdfB2;
    return denom > 0.0f ? pdfA2 / denom : 0.0f;
}

__host__ __device__ glm::vec3
calculateRandomDirectionInHemisphere(glm::vec3 normal,
                                     thrust::default_random_engine &rng) {
  thrust::uniform_real_distribution<float> u01(0, 1);

  float up = sqrt(u01(rng));      // cos(theta)
  float over = sqrt(1 - up * up); // sin(theta)
  float around = u01(rng) * TWO_PI;

  // Find a direction that is not the normal based off of whether or not the
  // normal's components are all equal to sqrt(1/3) or whether or not at
  // least one component is less than sqrt(1/3). Learned this trick from
  // Peter Kutz.

  glm::vec3 directionNotNormal;
  if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(1, 0, 0);
  } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = glm::vec3(0, 1, 0);
  } else {
    directionNotNormal = glm::vec3(0, 0, 1);
  }

  // Use not-normal direction to generate two perpendicular directions
  glm::vec3 perpendicularDirection1 =
      glm::normalize(glm::cross(normal, directionNotNormal));
  glm::vec3 perpendicularDirection2 =
      glm::normalize(glm::cross(normal, perpendicularDirection1));

  return up * normal + cos(around) * over * perpendicularDirection1 +
         sin(around) * over * perpendicularDirection2;
}

__device__ float shadow_intersect_aabb(const Ray& ray, const BVHNode& node,
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

    return tmax >= tmin && tmin < maxT ? tmin : FLT_MAX;
}

__device__ bool shadow_bvh_occluded(const Ray& shadowRay,
                                    Triangle* triangles,
                                    int triangles_size,
                                    BVHNode* bvh,
                                    float maxDistance) {
    if (triangles == nullptr || triangles_size <= 0 || bvh == nullptr) {
        return false;
    }

    int nodeIdx = 0;
    constexpr int STACK_SIZE = 128;
    int stack[STACK_SIZE];
    int stackPtr = 0;

    while (true) {
        BVHNode& node = bvh[nodeIdx];

        if (node.isLeaf()) {
            for (uint32_t i = 0; i < node.triCount; ++i) {
                int triangleIdx = (int)(node.firstTriIdx + i);
                if (triangleIdx >= triangles_size) {
                    continue;
                }

                glm::vec3 intersectPoint;
                glm::vec3 normal;
                bool outside = true;
                float t = triangleIntersectionTest(
                    triangles[triangleIdx], shadowRay, intersectPoint, normal,
                    outside);
                if (t > SHADOW_RAY_EPSILON && t < maxDistance) {
                    return true;
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
        float dist1 = shadow_intersect_aabb(shadowRay, bvh[child1],
                                            maxDistance);
        float dist2 = shadow_intersect_aabb(shadowRay, bvh[child2],
                                            maxDistance);

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

    return false;
}

__device__ bool shadow_ray_visible(const glm::vec3& origin,
                                   const NeeSample& neeSample,
                                   Geom* geoms,
                                   int geoms_size,
                                   Triangle* triangles,
                                   int triangles_size,
                                   BVHNode* bvh) {
    float maxDistance = neeSample.distance - 2.0f * SHADOW_RAY_EPSILON;
    if (maxDistance <= SHADOW_RAY_EPSILON) {
        return false;
    }

    Ray shadowRay;
    shadowRay.origin = origin + SHADOW_RAY_EPSILON * neeSample.wi;
    shadowRay.direction = neeSample.wi;

    for (int i = 0; i < geoms_size; ++i) {
        Geom& geom = geoms[i];
        glm::vec3 intersectPoint;
        glm::vec3 normal;
        bool outside = true;
        float t = -1.0f;
        if (geom.type == CUBE) {
            t = boxIntersectionTest(geom, shadowRay, intersectPoint, normal,
                                    outside);
        } else if (geom.type == SPHERE) {
            t = sphereIntersectionTest(geom, shadowRay, intersectPoint, normal,
                                       outside);
        }

        if (t > SHADOW_RAY_EPSILON && t < maxDistance) {
            return false;
        }
    }

    return !shadow_bvh_occluded(shadowRay, triangles, triangles_size, bvh,
                                maxDistance);
}

__device__ glm::vec3 evaluate_bsdf_for_direction(
    const Material& m,
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal) {
    if (m.is_metalic) {
        if (m.roughness_factor < 0.001f) {
            return glm::vec3(0.0f);
        }

        float alpha = fminf(fmaxf(m.roughness_factor, 0.001f), 1.0f);
#if REMAP_ROUGHNESS
        alpha = alpha * alpha;
#endif
        return Microfacet::evaluate_brdf(woWorld, wiWorld, normal, m, alpha);
    }

    if (glm::dot(normal, woWorld) <= 0.0f ||
        glm::dot(normal, wiWorld) <= 0.0f) {
        return glm::vec3(0.0f);
    }
    return m.color / PI;
}

__device__ float evaluate_bsdf_pdf_for_direction(
    const Material& m,
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal) {
    if (m.is_metalic) {
        if (m.roughness_factor < 0.001f) {
            return 0.0f;
        }

        float alpha = fminf(fmaxf(m.roughness_factor, 0.001f), 1.0f);
#if REMAP_ROUGHNESS
        alpha = alpha * alpha;
#endif
        return Microfacet::evaluate_pdf(woWorld, wiWorld, normal, alpha);
    }

    if (glm::dot(normal, woWorld) <= 0.0f ||
        glm::dot(normal, wiWorld) <= 0.0f) {
        return 0.0f;
    }
    return fmaxf(glm::dot(normal, wiWorld), 0.0f) / PI;
}

__device__ glm::vec3 evaluateEmissiveHit(
    const PathSegment& pathSegment,
    glm::vec3 hitPoint,
    glm::vec3 lightNormal,
    int triangleId,
    const Material& lightMaterial,
    SceneLight* lights,
    int lights_size,
    Triangle* triangles,
    int triangles_size) {
    float misWeight = 1.0f;
    if (!pathSegment.lastBounceWasSpecular &&
        pathSegment.lastBsdfPdf > 0.0f &&
        triangleId >= 0) {
        float lightPdf = Nee::pdf_light_for_triangle_hit(
            pathSegment.ray.origin, hitPoint, lightNormal, triangleId, lights,
            lights_size, triangles, triangles_size, lightMaterial);
        misWeight = power_heuristic(pathSegment.lastBsdfPdf, lightPdf);
    }

    return misWeight * lightMaterial.emissive_factor;
}

__device__ ScatterResult scatterRay(PathSegment &pathSegment,
                                    glm::vec3 intersect, glm::vec3 normal,
                                    const Material &m,
                                    SceneLight *lights,
                                    int lights_size,
                                    Geom *geoms,
                                    int geoms_size,
                                    Triangle *triangles,
                                    int triangles_size,
                                    BVHNode *bvh,
                                    Material *materials,
                                    int materials_size,
                                    thrust::default_random_engine &rng) {
    ScatterResult result;
    result.throughputMultiplier = glm::vec3(0.0f);
    result.contribution = glm::vec3(0.0f);
    result.pdf = 0.0f;
    result.wasSpecular = false;

    glm::vec3 woWorld = glm::normalize(-pathSegment.ray.direction);
    bool isPerfectSpecular = m.is_metalic && m.roughness_factor < 0.001f;
    if (!isPerfectSpecular) {
        NeeSample neeSample =
            Nee::get_nee(intersect, normal, lights, lights_size, triangles,
                         triangles_size, materials, materials_size, rng);
        if (neeSample.valid &&
            shadow_ray_visible(intersect, neeSample, geoms, geoms_size,
                               triangles, triangles_size, bvh)) {
            glm::vec3 bsdf =
                evaluate_bsdf_for_direction(m, woWorld, neeSample.wi, normal);
            float bsdfPdf =
                evaluate_bsdf_pdf_for_direction(m, woWorld, neeSample.wi,
                                                normal);
            float cosSurface = fmaxf(glm::dot(normal, neeSample.wi), 0.0f);
            if (cosSurface > 0.0f && neeSample.pdfLight > 0.0f) {
                float misWeight = power_heuristic(neeSample.pdfLight, bsdfPdf);
                result.contribution =
                    misWeight * neeSample.Li * bsdf * cosSurface /
                    neeSample.pdfLight;
            }
        }
    }

    if (m.is_metalic) {
        if (m.roughness_factor < 0.001) {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.ray.origin = intersect;
            result.throughputMultiplier = m.color;
            result.pdf = 1.0f;
            result.wasSpecular = true;
            return result;
        } else {
            float alpha = fminf(fmaxf(m.roughness_factor, 0.001f), 1.0f);
#if REMAP_ROUGHNESS
            alpha = alpha * alpha;
#endif

            ScatterResult sampledResult =
                Microfacet::get_brdf_result(pathSegment, intersect, normal, m,
                                            alpha, rng);
            sampledResult.contribution = result.contribution;
            return sampledResult;
        }

        pathSegment.ray.origin = intersect;
        result.throughputMultiplier = m.color;
        result.pdf = 1.0f;
        result.wasSpecular = true;
        return result;
    } else {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect;
        float cosTheta = fmaxf(glm::dot(normal, pathSegment.ray.direction), 0.0f);
        result.throughputMultiplier = m.color;
        result.pdf = cosTheta / PI;
        return result;
    }
}
