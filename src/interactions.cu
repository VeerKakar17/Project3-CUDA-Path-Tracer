#include "interactions.h"

#include "intersections.h"
#include "microfacet.h"
#include "nee.h"
#include "utilities.h"

#include <thrust/random.h>

#define REMAP_ROUGHNESS 1
#define SHADOW_RAY_EPSILON 0.0002f

__device__ float power_heuristic(float pdfA, float pdfB) {
    float pdfA2 = pdfA * pdfA;
    float pdfB2 = pdfB * pdfB;
    float denom = pdfA2 + pdfB2;
    return denom > 0.0f ? pdfA2 / denom : 0.0f;
}

__device__ float fresnel_dielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = fminf(fmaxf(cosThetaI, -1.0f), 1.0f);
    if (cosThetaI < 0.0f) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
        cosThetaI = -cosThetaI;
    }

    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.0f) {
        return 1.0f;
    }

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));
    float rParallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                      ((etaT * cosThetaI) + (etaI * cosThetaT));
    float rPerp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return 0.5f * (rParallel * rParallel + rPerp * rPerp);
}

__device__ float material_alpha(const Material& m) {
    float alpha = fminf(fmaxf(m.roughness_factor, 0.001f), 1.0f);
#if REMAP_ROUGHNESS
    alpha = alpha * alpha;
#endif
    return alpha;
}

__device__ void material_lobe_weights(const Material& m,
                                      float& wDiffuse,
                                      float& wMetal,
                                      float& wTransmission) {
    wMetal = m.metalic_factor;
    wTransmission = (1.0f - wMetal) * m.transmission_factor;
    wDiffuse = (1.0f - wMetal) * (1.0f - m.transmission_factor);
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
                                    Material* materials,
                                    int materials_size,
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
                int materialId = triangles[triangleIdx].materialid;
                if (materialId >= 0 && materialId < materials_size &&
                    materials[materialId].transmission_factor > 0.0f) {
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
                                   BVHNode* bvh,
                                   Material* materials,
                                   int materials_size) {
    float maxDistance = neeSample.distance - 2.0f * SHADOW_RAY_EPSILON;
    if (maxDistance <= SHADOW_RAY_EPSILON) {
        return false;
    }

    Ray shadowRay;
    shadowRay.origin = origin + SHADOW_RAY_EPSILON * neeSample.wi;
    shadowRay.direction = neeSample.wi;

    for (int i = 0; i < geoms_size; ++i) {
        Geom& geom = geoms[i];
        if (geom.materialid >= 0 && geom.materialid < materials_size &&
            materials[geom.materialid].transmission_factor > 0.0f) {
            continue;
        }
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
                                materials, materials_size, maxDistance);
}

__device__ glm::vec3 evaluate_bsdf_for_direction(
    const Material& m,
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    bool outside) {
    bool smooth = m.roughness_factor <= 0.001f;
    float wDiffuse;
    float wMetal;
    float wTransmission;
    material_lobe_weights(m, wDiffuse, wMetal, wTransmission);

    glm::vec3 value(0.0f);
    if (wDiffuse > 0.0f && glm::dot(normal, woWorld) > 0.0f &&
        glm::dot(normal, wiWorld) > 0.0f) {
        value += wDiffuse * m.color / PI;
    }

    if (!smooth && wMetal > 0.0f) {
        value += wMetal *
                 Microfacet::evaluate_brdf(woWorld, wiWorld, normal, m,
                                            material_alpha(m));
    }

    if (!smooth && wTransmission > 0.0f) {
        value += wTransmission *
                 Microfacet::evaluate_dielectric_bsdf(
                     woWorld, wiWorld, normal, m, outside,
                     material_alpha(m));
    }

    return value;
}

__device__ float evaluate_bsdf_pdf_for_direction(
    const Material& m,
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    bool outside) {
    bool smooth = m.roughness_factor <= 0.001f;
    float wDiffuse;
    float wMetal;
    float wTransmission;
    material_lobe_weights(m, wDiffuse, wMetal, wTransmission);

    float pdf = 0.0f;
    if (wDiffuse > 0.0f && glm::dot(normal, woWorld) > 0.0f &&
        glm::dot(normal, wiWorld) > 0.0f) {
        pdf += wDiffuse * fmaxf(glm::dot(normal, wiWorld), 0.0f) / PI;
    }

    if (!smooth && wMetal > 0.0f) {
        pdf += wMetal * Microfacet::evaluate_pdf(
                            woWorld, wiWorld, normal, material_alpha(m));
    }

    if (!smooth && wTransmission > 0.0f) {
        pdf += wTransmission *
               Microfacet::evaluate_dielectric_pdf(
                   woWorld, wiWorld, normal, m, outside,
                   material_alpha(m));
    }

    return pdf;
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
    int triangles_size,
    float totalLightArea) {
    float misWeight = 1.0f;
    if (!pathSegment.lastBounceWasSpecular &&
        pathSegment.lastBsdfPdf > 0.0f &&
        triangleId >= 0) {
        float lightPdf = Nee::pdf_light_for_triangle_hit(
            pathSegment.ray.origin, hitPoint, lightNormal, triangleId, lights,
            lights_size, triangles, triangles_size, lightMaterial,
            totalLightArea);
        misWeight = power_heuristic(pathSegment.lastBsdfPdf, lightPdf);
    }

    return misWeight * lightMaterial.emissive_factor;
}

__device__ ScatterResult empty_scatter_result() {
    ScatterResult result;
    result.throughputMultiplier = glm::vec3(0.0f);
    result.contribution = glm::vec3(0.0f);
    result.etaScale = 1.0f;
    result.pdf = 0.0f;
    result.wasSpecular = false;
    result.wasTransmission = false;
    return result;
}

__device__ ScatterResult calculate_reflection(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    bool pureDeltaMetal,
    const glm::vec3& sampledWm,
    thrust::default_random_engine& rng)
{
    ScatterResult result = empty_scatter_result();

    if (pureDeltaMetal) {
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect;
        result.throughputMultiplier = m.color;
        result.pdf = 1.0f;
        result.wasSpecular = true;
        return result;
    }

    float alpha = m.roughness_factor * m.roughness_factor;

    ScatterResult sampledResult =
        Microfacet::get_brdf_result(pathSegment, intersect, normal, m,
                                    alpha, sampledWm, rng);
    return sampledResult;
}

__device__ ScatterResult calculate_transmission(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& m,
    bool smooth,
    glm::vec3 woWorld,
    const glm::vec3& sampledWm,
    thrust::default_random_engine& rng)
{
    ScatterResult result = empty_scatter_result();
    glm::vec3 n = smooth ? normal : sampledWm;
    if (glm::dot(woWorld, n) < 0.0f) {
        n = -n;
    }

    glm::vec3 rayDir = glm::normalize(pathSegment.ray.direction);
    float ior = fmaxf(m.indexOfRefraction, 1.0001f);
    float etaI = outside ? 1.0f : ior;
    float etaT = outside ? ior : 1.0f;
    float etaP = etaT / etaI;

    float cosThetaO = fminf(fmaxf(glm::dot(woWorld, n), 0.0f), 1.0f);
    float R = fresnel_dielectric(cosThetaO, etaI, etaT);
    float T = 1.0f - R;

    thrust::uniform_real_distribution<float> u01(0, 1);
    float random = u01(rng);
    glm::vec3 wi;

    bool sampledReflection = random < R;
    if (sampledReflection) {
        wi = glm::reflect(rayDir, n);
        result.pdf = R;
    } else {
        wi = glm::refract(rayDir, n, etaI / etaT);
        if (glm::dot(wi, wi) <= 0.0f) {
            return result;
        }
        result.pdf = T;
        result.etaScale = etaP * etaP;
        result.wasTransmission = true;
    }

    if (!smooth) {
        glm::vec3 wiWorld = glm::normalize(wi);
        float macroCosThetaI = glm::dot(normal, wiWorld);
        if ((sampledReflection && macroCosThetaI <= 0.0f) ||
            (!sampledReflection && macroCosThetaI >= 0.0f)) {
            return result;
        }

        glm::vec3 bsdf = Microfacet::evaluate_dielectric_bsdf(
            woWorld, wiWorld, normal, m, outside, material_alpha(m));
        float pdf = Microfacet::evaluate_dielectric_pdf(
            woWorld, wiWorld, normal, m, outside, material_alpha(m));
        if (pdf <= 0.0f) {
            return result;
        }

        float cosTheta = fabsf(glm::dot(normal, wiWorld));
        result.throughputMultiplier = bsdf * cosTheta / pdf;
        result.pdf = pdf;
    } else {
        result.throughputMultiplier = m.color;
        if (random >= R) {
            result.throughputMultiplier /= etaP * etaP;
        }
    }

    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.ray.origin =
        intersect + SHADOW_RAY_EPSILON * pathSegment.ray.direction;
    result.wasSpecular = smooth;

    return result;
}

__device__ ScatterResult calculate_diffuse(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    ScatterResult result = empty_scatter_result();
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.ray.origin = intersect;
    float cosTheta = fmaxf(glm::dot(normal, pathSegment.ray.direction), 0.0f);
    result.throughputMultiplier = m.color;
    result.pdf = cosTheta / PI;
    return result;
}

__device__ glm::vec3 get_volume_transmittance(const glm::vec3& attenuationColor,
                                              float attenuationDistance,
                                              float distance) {
    if (attenuationDistance <= 0.0f || attenuationDistance >= FLT_MAX ||
        distance <= 0.0f) {
        return glm::vec3(1.0f);
    }

    float x = distance / attenuationDistance;
    return glm::pow(attenuationColor, glm::vec3(x));
}

__device__ ScatterResult scatterRay(PathSegment &pathSegment,
                                    glm::vec3 intersect, glm::vec3 normal,
                                    bool outside,
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
                                    int materialId,
                                    float totalLightArea,
                                    thrust::default_random_engine &rng) {
    const float materialEpsilon = 0.001f;
    (void)totalLightArea;
    
    ScatterResult result = empty_scatter_result();

    glm::vec3 woWorld = glm::normalize(-pathSegment.ray.direction);
    
    bool smooth =
        m.roughness_factor <= materialEpsilon;
    float alpha = material_alpha(m);

    bool pureMetal =
        m.metalic_factor >= 1.0f - materialEpsilon;

    bool pureGlass =
        m.metalic_factor <= materialEpsilon &&
        m.transmission_factor >= 1.0f - materialEpsilon;

    bool pureDeltaMetal =
        pureMetal && smooth;

    bool pureDeltaGlass =
        pureGlass && smooth;

    bool doMIS =
        !(pureDeltaMetal || pureDeltaGlass);

    glm::vec3 sampledWm = normal;
    if (!smooth && (m.metalic_factor > materialEpsilon ||
                    m.transmission_factor > materialEpsilon)) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec2 u(u01(rng), u01(rng));
        sampledWm = Microfacet::sample_wm(alpha, normal, u);
    }

    // determine if do NEE + MIS if pure delta
    if (doMIS) {
        NeeSample neeSample =
        Nee::get_nee(intersect, normal, lights, lights_size, triangles,
                         triangles_size, materials, materials_size,
                         totalLightArea, rng);
        if (neeSample.valid &&
            shadow_ray_visible(intersect, neeSample, geoms, geoms_size,
                               triangles, triangles_size, bvh, materials,
                               materials_size)) {
            glm::vec3 bsdf =
                evaluate_bsdf_for_direction(m, woWorld, neeSample.wi, normal,
                                            outside);
            float bsdfPdf =
                evaluate_bsdf_pdf_for_direction(m, woWorld, neeSample.wi,
                                                normal, outside);
            float cosSurface = fmaxf(glm::dot(normal, neeSample.wi), 0.0f);
            if (cosSurface > 0.0f && neeSample.pdfLight > 0.0f) {
                float misWeight = power_heuristic(neeSample.pdfLight, bsdfPdf);
                result.contribution =
                    misWeight * neeSample.Li * bsdf * cosSurface /
                    neeSample.pdfLight;
            }
        }
    }

    // Case pure metal
    if (pureMetal) {
        ScatterResult sampledResult =
            calculate_reflection(pathSegment, intersect, normal, m,
                                 pureDeltaMetal, sampledWm, rng);
        sampledResult.contribution = result.contribution;
        return sampledResult;
    }

    // Case pure dielectric
    if (pureGlass) {
        ScatterResult sampledResult =
            calculate_transmission(pathSegment, intersect, normal, outside, m,
                                   smooth, woWorld, sampledWm, rng);
        sampledResult.contribution = result.contribution;

        if (sampledResult.wasTransmission) {
            if (outside && m.thicknessFactor > 0.0f) {
                pathSegment.mediumMaterialIndex = materialId;
            } else if (!outside) {
                pathSegment.mediumMaterialIndex = -1;
            }
        }

        return sampledResult;
    }

    if (m.transmission_factor < materialEpsilon &&
        m.metalic_factor < materialEpsilon) {
        ScatterResult sampledResult =
            calculate_diffuse(pathSegment, intersect, normal, m, rng);
        sampledResult.contribution = result.contribution;
        return sampledResult;
    }

    // Opaque metalic + dielectric mix
    float p_diffuse;
    float p_metal;
    float p_transmission;
    material_lobe_weights(m, p_diffuse, p_metal, p_transmission);

    thrust::uniform_real_distribution<float> u01(0, 1);
    float random_chance = u01(rng);

    ScatterResult sampledResult = empty_scatter_result();
    if (random_chance < p_metal) {
        sampledResult =
            calculate_reflection(pathSegment, intersect, normal, m,
                                 pureDeltaMetal, sampledWm, rng);
    } else if (random_chance < p_metal + p_transmission) {
        sampledResult =
            calculate_transmission(pathSegment, intersect, normal, outside, m,
                                   smooth, woWorld, sampledWm, rng);
    } else {
        sampledResult = calculate_diffuse(pathSegment, intersect, normal, m, rng);
    }

    if (!sampledResult.wasSpecular) {
        glm::vec3 wiWorld = glm::normalize(pathSegment.ray.direction);
        float pdf =
            evaluate_bsdf_pdf_for_direction(m, woWorld, wiWorld, normal,
                                            outside);
        glm::vec3 bsdf =
            evaluate_bsdf_for_direction(m, woWorld, wiWorld, normal, outside);

        if (pdf <= 0.0f ||
            (bsdf.x == 0.0f && bsdf.y == 0.0f && bsdf.z == 0.0f)) {
            sampledResult = empty_scatter_result();
            sampledResult.contribution = result.contribution;
            return sampledResult;
        }

        sampledResult.pdf = pdf;
        sampledResult.throughputMultiplier =
            bsdf * fabsf(glm::dot(normal, wiWorld)) / pdf;
    }

    if (sampledResult.wasTransmission) {
        if (outside && m.thicknessFactor > 0.0f) {
            pathSegment.mediumMaterialIndex = materialId;
        } else if (!outside) {
            pathSegment.mediumMaterialIndex = -1;
        }
    }

    sampledResult.contribution = result.contribution;
    return sampledResult;
}
