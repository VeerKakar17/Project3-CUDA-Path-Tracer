#pragma once

#include "interactions.h"

namespace Microfacet
{
__host__ __device__ ScatterResult get_brdf_result(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& material,
    float alpha,
    thrust::default_random_engine& rng);

__host__ __device__ glm::vec3 evaluate_brdf(
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    const Material& material,
    float alpha);

__host__ __device__ float evaluate_pdf(
    const glm::vec3& woWorld,
    const glm::vec3& wiWorld,
    const glm::vec3& normal,
    float alpha);
}
