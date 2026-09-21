#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "tinygltf/tiny_gltf_v3.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

static glm::vec3 readVec3(const json& value)
{
    return glm::vec3((float)value[0], (float)value[1], (float)value[2]);
}

static void finishMaterial(Material& material)
{
    material.is_metalic = material.metalic_factor > 0.0f ? 1 : 0;
    material.is_emissive =
        glm::dot(material.emissive_factor, material.emissive_factor) > 0.0f
            ? 1
            : 0;
    material.specular.color =
        glm::mix(glm::vec3(0.04f), material.color, material.metalic_factor);
    material.specular.exponent =
        (1.0f - material.roughness_factor) * 256.0f + 1.0f;
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    /*
    else if (ext == ".gltf" || ext == ".glb")
    {
        loadFromGltf(filename);
        return;
    } */
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        const auto& col = p["RGB"];
        newMaterial.color = readVec3(col);
        newMaterial.alpha = p.value("ALPHA", 1.0f);
        newMaterial.metalic_factor = p.value("METALLIC", 0.0f);
        newMaterial.roughness_factor = p.value("ROUGHNESS", 1.0f);
        newMaterial.emissive_factor = glm::vec3(0.0f);
        newMaterial.double_sided = p.value("DOUBLE_SIDED", false) ? 1 : 0;
        newMaterial.hasRefractive = 0;
        newMaterial.indexOfRefraction = 1.0f;

        if (p["TYPE"] == "Diffuse")
        {
        }
        else if (p["TYPE"] == "Emitting")
        {
            if (!p.contains("EMISSIVE_FACTOR"))
            {
                newMaterial.emissive_factor = newMaterial.color;
            }
        }
        else if (p["TYPE"] == "Specular")
        {
            if (!p.contains("METALLIC"))
            {
                newMaterial.metalic_factor = 1.0f;
            }
            if (!p.contains("ROUGHNESS"))
            {
                newMaterial.roughness_factor = 0.0f;
            }
        }

        if (p.contains("EMISSIVE_FACTOR"))
        {
            newMaterial.emissive_factor = readVec3(p["EMISSIVE_FACTOR"]);
        }

        finishMaterial(newMaterial);
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

/*
static int32_t findAttribute(
    const tg3_primitive& primitive,
    const char* name)
{
    const size_t nameLen = strlen(name);

    for (uint32_t i = 0; i < primitive.attributes_count; ++i)
    {
        const tg3_str_int_pair& attr = primitive.attributes[i];

        if (attr.key.len == nameLen &&
            memcmp(attr.key.data, name, nameLen) == 0)
        {
            return attr.value;
        }
    }

    return -1;
}

static void parse_materials(const tg3_model &model, std::vector<Material> &materials) {
    materials.clear();
    materials.reserve(model.materials_count);

    for (uint32_t i = 0; i < model.materials_count; i++) {
        const tg3_material &mat = model.materials[i];
        const tg3_pbr_metallic_roughness &pbr = mat.pbr_metallic_roughness;

        Material newMaterial{};
        newMaterial.color = glm::vec3(
            (float)pbr.base_color_factor[0],
            (float)pbr.base_color_factor[1],
            (float)pbr.base_color_factor[2]);
        newMaterial.alpha = (float)pbr.base_color_factor[3];

        newMaterial.metalic_factor = (float)pbr.metallic_factor;
        newMaterial.roughness_factor = (float)pbr.roughness_factor;
        newMaterial.is_metalic = newMaterial.metalic_factor > 0.0f ? 1 : 0;

        newMaterial.specular.color =
            glm::mix(glm::vec3(0.04f), newMaterial.color,
                     newMaterial.metalic_factor);
        newMaterial.specular.exponent =
            (1.0f - newMaterial.roughness_factor) * 256.0f + 1.0f;

        newMaterial.emissive_factor = glm::vec3(
            (float)mat.emissive_factor[0],
            (float)mat.emissive_factor[1],
            (float)mat.emissive_factor[2]);
        newMaterial.is_emissive =
            glm::dot(newMaterial.emissive_factor,
                     newMaterial.emissive_factor) > 0.0f
                ? 1
                : 0;

        newMaterial.double_sided = mat.double_sided ? 1 : 0;
        newMaterial.hasRefractive = 0;
        newMaterial.indexOfRefraction = 1.0f;

        materials.emplace_back(newMaterial);
    }
}

void Scene::loadFromGltf(const std::string &gltfName) {
    tg3_parse_options opts;
    tg3_error_stack errors;
    tg3_model model;

    tg3_parse_options_init(&opts);
    tg3_error_stack_init(&errors);

    tg3_error_code err = tg3_parse_file(&model, &errors, gltfName.c_str(),
                                        (uint32_t)gltfName.size(), &opts);
    if (err != TG3_OK) {
        for (uint32_t i = 0; i < errors.count; i++) {
            fprintf(stderr, "[%d] %s\n", (int)errors.entries[i].severity,
                    errors.entries[i].message ? errors.entries[i].message : "(null)");
        }
        tg3_error_stack_free(&errors);
        exit(-1);
    }
    
    parse_materials(model, materials);
    
    tg3_model_free(&model);
    tg3_error_stack_free(&errors);
}

/*
typedef struct tg3_model {
    struct tg3_arena *arena_;  /* Internal, all memory owned here /

    const tg3_accessor      *accessors;      uint32_t accessors_count;
    const tg3_animation     *animations;     uint32_t animations_count;
    const tg3_buffer        *buffers;        uint32_t buffers_count;
    const tg3_buffer_view   *buffer_views;   uint32_t buffer_views_count;
    const tg3_material      *materials;      uint32_t materials_count;
    const tg3_mesh          *meshes;         uint32_t meshes_count;
    const tg3_node          *nodes;          uint32_t nodes_count;
    const tg3_texture       *textures;       uint32_t textures_count;
    const tg3_image         *images;         uint32_t images_count;
    const tg3_skin          *skins;          uint32_t skins_count;
    const tg3_sampler       *samplers;       uint32_t samplers_count;
    const tg3_camera        *cameras;        uint32_t cameras_count;
    const tg3_scene         *scenes;         uint32_t scenes_count;
    const tg3_light         *lights;         uint32_t lights_count;
    const tg3_audio_emitter *audio_emitters; uint32_t audio_emitters_count;
    const tg3_audio_source  *audio_sources;  uint32_t audio_sources_count;

    int32_t    default_scene;
    const tg3_str *extensions_used;      uint32_t extensions_used_count;
    const tg3_str *extensions_required;  uint32_t extensions_required_count;
    tg3_asset  asset;
    tg3_extras_ext ext;
} tg3_model; 
*/
