#pragma once

#include "sceneStructs.h"
#include <vector>
#include "bvh.h"

class BVH;

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGltf(const std::string& gltfName);
    void buildLightList();
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::vector<SceneLight> lights;
    BVH bvh;
    RenderState state;
};
