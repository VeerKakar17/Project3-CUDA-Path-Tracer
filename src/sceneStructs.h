#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <cstdint>
#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Triangle {
    glm::vec3 v0, v1, v2;
    glm::vec3 n0, n1, n2;
    glm::vec3 t0, t1, t2;
    glm::vec2 uv0, uv1, uv2;
    glm::vec3 centroid;
    int materialid;
};

struct Material
{
    glm::vec3 color;
    float alpha;

    uint8_t is_metalic;
    float metalic_factor;
    float roughness_factor;

    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;

    uint8_t is_emissive;
    glm::vec3 emissive_factor; // Emission color

    uint8_t double_sided; // For if we cull back faces

    uint8_t hasRefractive;
    float indexOfRefraction;

    int baseColorTexId;
    int metallicRoughnessTexId;
    int emissiveTexId;
    int normalTexId;

    Material() : emissiveTexId(-1), normalTexId(-1), baseColorTexId(-1), metallicRoughnessTexId(-1),
        hasRefractive(0), is_emissive(0), is_metalic(0) {}
};

struct Texture {
    int width;
    int height;
    int channels;
    std::vector<uchar4> pixels;
};

struct DeviceTexture {
    int width;
    int height;
    int channels;
    uchar4 *pixels;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 surfaceTangent;
  int materialId;
  int geomId;
  glm::vec2 uv;
};
