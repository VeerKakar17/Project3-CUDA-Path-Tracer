#include "nee.h"

namespace Nee
{

__device__ LightSample sample_light(
    SceneLight* lights,
    int lights_size,
    thrust::default_random_engine& rng)
{
    LightSample sample{};
    sample.lightIndex = -1;
    sample.lightPickPdf = 0.0f;
    sample.valid = false;

    if (lights == nullptr || lights_size <= 0)
    {
        return sample;
    }

    int triangleLightCount = 0;
    for (int i = 0; i < lights_size; ++i)
    {
        if (lights[i].type == SCENE_LIGHT_TRIANGLE)
        {
            ++triangleLightCount;
        }
    }
    if (triangleLightCount == 0)
    {
        return sample;
    }

    thrust::uniform_int_distribution<int> pickLight(0, triangleLightCount - 1);
    int selectedTriangleLight = pickLight(rng);
    int seenTriangleLights = 0;
    for (int i = 0; i < lights_size; ++i)
    {
        if (lights[i].type != SCENE_LIGHT_TRIANGLE)
        {
            continue;
        }
        if (seenTriangleLights == selectedTriangleLight)
        {
            sample.light = lights[i];
            sample.lightIndex = i;
            sample.lightPickPdf = 1.0f / (float)triangleLightCount;
            sample.valid = true;
            return sample;
        }
        ++seenTriangleLights;
    }

    return sample;
}

__device__ LightPointSample sample_point_from_light(
    const LightSample& lightSample,
    Triangle* triangles,
    int triangles_size,
    thrust::default_random_engine& rng)
{
    LightPointSample sample{};
    sample.materialId = -1;
    sample.pdfArea = 0.0f;
    sample.valid = false;

    if (!lightSample.valid ||
        lightSample.light.type != SCENE_LIGHT_TRIANGLE ||
        triangles == nullptr ||
        lightSample.light.id < 0 ||
        lightSample.light.id >= triangles_size)
    {
        return sample;
    }

    const Triangle& triangle = triangles[lightSample.light.id];
    glm::vec3 edge1 = triangle.v1 - triangle.v0;
    glm::vec3 edge2 = triangle.v2 - triangle.v0;
    glm::vec3 geometricNormal = glm::cross(edge1, edge2);
    float normalLength = glm::length(geometricNormal);
    if (normalLength <= 0.0f)
    {
        return sample;
    }

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float r1 = sqrtf(u01(rng));
    float r2 = u01(rng);
    float b0 = 1.0f - r1;
    float b1 = r1 * (1.0f - r2);
    float b2 = r1 * r2;

    sample.position = b0 * triangle.v0 + b1 * triangle.v1 + b2 * triangle.v2;
    sample.normal = b0 * triangle.n0 + b1 * triangle.n1 + b2 * triangle.n2;
    if (glm::dot(sample.normal, sample.normal) <= 0.0f)
    {
        sample.normal = geometricNormal;
    }
    sample.normal = glm::normalize(sample.normal);
    sample.barycentric = glm::vec3(b0, b1, b2);
    sample.materialId = triangle.materialid;
    sample.pdfArea = 2.0f / normalLength;
    sample.valid = true;
    return sample;
}

__device__ NeeSample get_nee(
    const glm::vec3& surfacePoint,
    const glm::vec3& surfaceNormal,
    SceneLight* lights,
    int lights_size,
    Triangle* triangles,
    int triangles_size,
    Material* materials,
    int materials_size,
    thrust::default_random_engine& rng)
{
    NeeSample result{};
    result.distance = 0.0f;
    result.pdfArea = 0.0f;
    result.pdfDirectional = 0.0f;
    result.pdfLight = 0.0f;
    result.lightIndex = -1;
    result.materialId = -1;
    result.valid = false;

    LightSample lightSample = sample_light(lights, lights_size, rng);
    LightPointSample pointSample =
        sample_point_from_light(lightSample, triangles, triangles_size, rng);
    if (!lightSample.valid || !pointSample.valid ||
        pointSample.materialId < 0 || pointSample.materialId >= materials_size ||
        materials == nullptr)
    {
        return result;
    }

    glm::vec3 toLight = pointSample.position - surfacePoint;
    float distanceSquared = glm::dot(toLight, toLight);
    if (distanceSquared <= 0.0f)
    {
        return result;
    }

    float distance = sqrtf(distanceSquared);
    glm::vec3 wi = toLight / distance;
    float cosLight = glm::dot(pointSample.normal, -wi);
    const Material& lightMaterial = materials[pointSample.materialId];
    if (lightMaterial.double_sided)
    {
        cosLight = fabsf(cosLight);
    }
    else
    {
        cosLight = fmaxf(cosLight, 0.0f);
    }
    if (cosLight <= 0.0f || lightSample.lightPickPdf <= 0.0f ||
        pointSample.pdfArea <= 0.0f)
    {
        return result;
    }

    float cosSurface = fmaxf(glm::dot(surfaceNormal, wi), 0.0f);
    if (cosSurface <= 0.0f)
    {
        return result;
    }

    result.position = pointSample.position;
    result.normal = pointSample.normal;
    result.wi = wi;
    result.Li = lightMaterial.emissive_factor;
    result.distance = distance;
    result.pdfArea = pointSample.pdfArea;
    result.pdfDirectional = pointSample.pdfArea * distanceSquared / cosLight;
    result.pdfLight = lightSample.lightPickPdf * result.pdfDirectional;
    result.lightIndex = lightSample.lightIndex;
    result.materialId = pointSample.materialId;
    result.valid = result.pdfLight > 0.0f &&
                   glm::dot(result.Li, result.Li) > 0.0f;
    return result;
}

__device__ float pdf_light_for_triangle_hit(
    const glm::vec3& surfacePoint,
    const glm::vec3& lightPoint,
    const glm::vec3& lightNormal,
    int triangleId,
    SceneLight* lights,
    int lights_size,
    Triangle* triangles,
    int triangles_size,
    const Material& lightMaterial)
{
    if (triangleId < 0 || triangleId >= triangles_size ||
        triangles == nullptr || lights == nullptr || lights_size <= 0)
    {
        return 0.0f;
    }

    int triangleLightCount = 0;
    bool selectedTriangleIsLight = false;
    for (int i = 0; i < lights_size; ++i)
    {
        if (lights[i].type != SCENE_LIGHT_TRIANGLE)
        {
            continue;
        }
        ++triangleLightCount;
        if (lights[i].id == triangleId)
        {
            selectedTriangleIsLight = true;
        }
    }
    if (!selectedTriangleIsLight || triangleLightCount == 0)
    {
        return 0.0f;
    }

    const Triangle& triangle = triangles[triangleId];
    glm::vec3 edge1 = triangle.v1 - triangle.v0;
    glm::vec3 edge2 = triangle.v2 - triangle.v0;
    float normalLength = glm::length(glm::cross(edge1, edge2));
    if (normalLength <= 0.0f)
    {
        return 0.0f;
    }

    glm::vec3 toLight = lightPoint - surfacePoint;
    float distanceSquared = glm::dot(toLight, toLight);
    if (distanceSquared <= 0.0f)
    {
        return 0.0f;
    }

    glm::vec3 wi = glm::normalize(toLight);
    float cosLight = glm::dot(lightNormal, -wi);
    cosLight = lightMaterial.double_sided ? fabsf(cosLight)
                                          : fmaxf(cosLight, 0.0f);
    if (cosLight <= 0.0f)
    {
        return 0.0f;
    }

    float pdfArea = 2.0f / normalLength;
    float pdfDirectional = pdfArea * distanceSquared / cosLight;
    float lightPickPdf = 1.0f / (float)triangleLightCount;
    return lightPickPdf * pdfDirectional;
}

}
