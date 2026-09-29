#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

#define REMAP_ROUGHNESS 0

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

__host__ __device__ float abs_cos_theta(const glm::vec3 &w) {
    return fabsf(w.z);
}

__host__ __device__ bool same_hemisphere(const glm::vec3 &a,
                                         const glm::vec3 &b) {
    return a.z * b.z > 0.0f;
}

__host__ __device__ glm::vec3 to_local(const glm::vec3 &v,
                                       const glm::vec3 &tangent,
                                       const glm::vec3 &bitangent,
                                       const glm::vec3 &normal) {
    return glm::vec3(glm::dot(v, tangent), glm::dot(v, bitangent),
                     glm::dot(v, normal));
}

__host__ __device__ glm::vec3 to_world(const glm::vec3 &v,
                                       const glm::vec3 &tangent,
                                       const glm::vec3 &bitangent,
                                       const glm::vec3 &normal) {
    return v.x * tangent + v.y * bitangent + v.z * normal;
}

__host__ __device__ void make_basis(const glm::vec3 &normal,
                                    glm::vec3 &tangent,
                                    glm::vec3 &bitangent) {
    glm::vec3 helper = fabsf(normal.x) < SQRT_OF_ONE_THIRD
                           ? glm::vec3(1.0f, 0.0f, 0.0f)
                           : glm::vec3(0.0f, 1.0f, 0.0f);
    tangent = glm::normalize(glm::cross(helper, normal));
    bitangent = glm::cross(normal, tangent);
}

__host__ __device__ float get_d_ggx(const glm::vec3 &m, float alpha) {
    float cosTheta = abs_cos_theta(m);
    if (cosTheta <= 0.0f) {
        return 0.0f;
    }

    float alpha2 = alpha * alpha;
    float cosTheta2 = cosTheta * cosTheta;
    float inner = cosTheta2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * inner * inner);
}

__host__ __device__ float get_pdf(const glm::vec3 &wo,
                                  const glm::vec3 &wm,
                                  float alpha) {
    float woDotWm = fabsf(glm::dot(wo, wm));
    if (woDotWm <= 0.0f) {
        return 0.0f;
    }
    return get_d_ggx(wm, alpha) * abs_cos_theta(wm) / (4.0f * woDotWm);
}

__host__ __device__ float get_g_lambda(const glm::vec3 &w, float alpha) {
    float cosTheta = abs_cos_theta(w);
    if (cosTheta <= 0.0f) {
        return 0.0f;
    }

    float cosTheta2 = cosTheta * cosTheta;
    float sinTheta2 = fmaxf(0.0f, 1.0f - cosTheta2);
    float tanTheta2 = sinTheta2 / cosTheta2;
    return 0.5f * (sqrtf(1.0f + alpha * alpha * tanTheta2) - 1.0f);
}

__host__ __device__ float get_g1(const glm::vec3 &w, float alpha) {
    return 1.0f / (1.0f + get_g_lambda(w, alpha));
}

__host__ __device__ float get_g_ggx(const glm::vec3 &wi,
                                    const glm::vec3 &wo,
                                    float alpha) {
    return 1.0f /
           (1.0f + get_g_lambda(wo, alpha) + get_g_lambda(wi, alpha));
}

__host__ __device__ float get_f_ggx(const glm::vec3 &wi,
                                    const glm::vec3 &wo,
                                    float alpha) {
    const float f0 = 0.04f;
    glm::vec3 wm = glm::normalize(wi + wo);
    float cosTheta = fminf(fmaxf(fabsf(glm::dot(wi, wm)), 0.0f), 1.0f);
    float oneMinusCos = 1.0f - cosTheta;
    float oneMinusCos2 = oneMinusCos * oneMinusCos;
    return f0 + (1.0f - f0) * oneMinusCos2 * oneMinusCos2 * oneMinusCos;
}

__host__ __device__ glm::vec3 sample_wm(float alpha, const glm::vec2 &u) {
    float phi = TWO_PI * u.x;
    float alpha2 = alpha * alpha;
    float cosTheta =
        sqrtf((1.0f - u.y) / (1.0f + (alpha2 - 1.0f) * u.y));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    return glm::normalize(glm::vec3(cosf(phi) * sinTheta,
                                    sinf(phi) * sinTheta,
                                    cosTheta));
}

// returns multiplier for color
__host__ __device__ float scatterRay(PathSegment &pathSegment,
                                    glm::vec3 intersect, glm::vec3 normal,
                                    const Material &m,
                                    thrust::default_random_engine &rng) {
    if (m.is_metalic) {
        if (m.roughness_factor < 0.001) {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.ray.origin = intersect;
            return 1.0f;
        } else {
            float alpha = fminf(fmaxf(m.roughness_factor, 0.001f), 1.0f);
#if REMAP_ROUGHNESS
            alpha = alpha * alpha;
#endif

            thrust::uniform_real_distribution<float> u01(0, 1);
            glm::vec2 u(u01(rng), u01(rng));

            glm::vec3 tangent;
            glm::vec3 bitangent;
            make_basis(normal, tangent, bitangent);

            glm::vec3 woWorld = glm::normalize(-pathSegment.ray.direction);
            glm::vec3 wo = to_local(woWorld, tangent, bitangent, normal);
            if (wo.z <= 0.0f) {
                pathSegment.ray.direction =
                    glm::reflect(pathSegment.ray.direction, normal);
                pathSegment.ray.origin = intersect;
                return 1.0f;
            }

            glm::vec3 wm = sample_wm(alpha, u);
            glm::vec3 wi = glm::reflect(-wo, wm);
            if (!same_hemisphere(wo, wi) || wi.z <= 0.0f) {
                pathSegment.ray.direction =
                    calculateRandomDirectionInHemisphere(normal, rng);
                pathSegment.ray.origin = intersect;
                return 0.0f;
            }

            pathSegment.ray.direction =
                glm::normalize(to_world(wi, tangent, bitangent, normal));
            pathSegment.ray.origin = intersect;

            float pdf = get_pdf(wo, wm, alpha);
            if (pdf <= 0.0f) {
                return 0.0f;
            }

            float cosThetaO = abs_cos_theta(wo);
            float cosThetaI = abs_cos_theta(wi);
            float d = get_d_ggx(wm, alpha);
            float g = get_g_ggx(wi, wo, alpha);
            float f = get_f_ggx(wi, wo, alpha);
            float brdf = d * g * f / (4.0f * cosThetaI * cosThetaO);

            return brdf * cosThetaI / pdf;
        }

        pathSegment.ray.origin = intersect;
        return 1.0f;
    } else {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect;
        return 1.0f;
    }
    
}
