#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "tinygltf/tiny_gltf_v3.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <string>
#include <unordered_map>

#define TINYGLTF3_ENABLE_FS 1

using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;

static void appendGltfFile(const std::string& gltfName,
                           const glm::mat4& rootTransform,
                           std::vector<Material>& materials,
                           std::vector<Triangle>& triangles);

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

static void finalizeCamera(RenderState& state, float fovy)
{
    Camera& camera = state.camera;
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
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
    else if (ext == ".gltf" || ext == ".glb")
    {
        loadFromGltf(filename);
        return;
    }
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
    fs::path jsonDir = fs::path(jsonName).parent_path();
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
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];

        if (type == "mesh")
        {
            fs::path meshPath = p["FILE"].get<std::string>();
            if (meshPath.is_relative())
            {
                meshPath = jsonDir / meshPath;
            }

            glm::mat4 rootTransform = utilityCore::buildTransformationMatrix(
                readVec3(trans), readVec3(rotat), readVec3(scale));
            appendGltfFile(meshPath.string(), rootTransform, materials,
                           triangles);
            continue;
        }

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

    finalizeCamera(state, fovy);
}

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

static glm::mat4 readGltfMatrix(const double matrix[16])
{
    glm::mat4 result(1.0f);
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            result[col][row] = (float)matrix[col * 4 + row];
        }
    }
    return result;
}

static glm::mat4 getNodeLocalTransform(const tg3_node& node)
{
    if (node.has_matrix)
    {
        return readGltfMatrix(node.matrix);
    }

    glm::mat4 translation = glm::translate(
        glm::mat4(1.0f),
        glm::vec3((float)node.translation[0], (float)node.translation[1],
                  (float)node.translation[2]));
    glm::quat rotation((float)node.rotation[3], (float)node.rotation[0],
                       (float)node.rotation[1], (float)node.rotation[2]);
    glm::mat4 scale = glm::scale(
        glm::mat4(1.0f),
        glm::vec3((float)node.scale[0], (float)node.scale[1],
                  (float)node.scale[2]));

    return translation * glm::mat4_cast(rotation) * scale;
}

static int append_materials(const tg3_model &model, std::vector<Material> &materials) {
    int materialOffset = (int)materials.size();
    uint32_t materialCount = model.materials_count > 0 ? model.materials_count : 1u;
    materials.reserve(materials.size() + materialCount);

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

        newMaterial.emissive_factor = glm::vec3(
            (float)mat.emissive_factor[0],
            (float)mat.emissive_factor[1],
            (float)mat.emissive_factor[2]);

        newMaterial.double_sided = mat.double_sided ? 1 : 0;
        newMaterial.hasRefractive = 0;
        newMaterial.indexOfRefraction = 1.0f;

        finishMaterial(newMaterial);
        materials.emplace_back(newMaterial);
    }

    if (model.materials_count == 0)
    {
        Material defaultMaterial{};
        defaultMaterial.color = glm::vec3(1.0f);
        defaultMaterial.alpha = 1.0f;
        defaultMaterial.metalic_factor = 0.0f;
        defaultMaterial.roughness_factor = 1.0f;
        defaultMaterial.emissive_factor = glm::vec3(0.0f);
        defaultMaterial.double_sided = 0;
        defaultMaterial.hasRefractive = 0;
        defaultMaterial.indexOfRefraction = 1.0f;
        finishMaterial(defaultMaterial);
        materials.emplace_back(defaultMaterial);
    }

    return materialOffset;
}

static const uint8_t* getAccessorElementPtr(const tg3_model& model,
                                            const tg3_accessor& accessor,
                                            uint64_t elementIndex,
                                            int32_t& stride)
{
    if (accessor.buffer_view < 0 ||
        accessor.buffer_view >= (int32_t)model.buffer_views_count ||
        elementIndex >= accessor.count)
    {
        return nullptr;
    }

    const tg3_buffer_view& bufferView = model.buffer_views[accessor.buffer_view];
    if (bufferView.buffer < 0 ||
        bufferView.buffer >= (int32_t)model.buffers_count)
    {
        return nullptr;
    }

    stride = tg3_accessor_byte_stride(&accessor, &bufferView);
    if (stride <= 0)
    {
        return nullptr;
    }

    const tg3_buffer& buffer = model.buffers[bufferView.buffer];
    uint64_t byteOffset =
        bufferView.byte_offset + accessor.byte_offset + elementIndex * stride;
    if (byteOffset + stride > buffer.data.count)
    {
        return nullptr;
    }

    return buffer.data.data + byteOffset;
}

static bool readAccessorVec3(const tg3_model& model, int32_t accessorIndex,
                             uint64_t elementIndex, glm::vec3& out)
{
    if (accessorIndex < 0 || accessorIndex >= (int32_t)model.accessors_count)
    {
        return false;
    }

    const tg3_accessor& accessor = model.accessors[accessorIndex];
    if (accessor.component_type != TG3_COMPONENT_TYPE_FLOAT ||
        accessor.type != TG3_TYPE_VEC3 || accessor.sparse.is_sparse)
    {
        return false;
    }

    int32_t stride = 0;
    const uint8_t* ptr =
        getAccessorElementPtr(model, accessor, elementIndex, stride);
    if (ptr == nullptr)
    {
        return false;
    }

    float values[3];
    memcpy(values, ptr, sizeof(values));
    out = glm::vec3(values[0], values[1], values[2]);
    return true;
}

static bool readAccessorIndex(const tg3_model& model, int32_t accessorIndex,
                              uint64_t elementIndex, uint32_t& out)
{
    if (accessorIndex < 0 || accessorIndex >= (int32_t)model.accessors_count)
    {
        return false;
    }

    const tg3_accessor& accessor = model.accessors[accessorIndex];
    if (accessor.type != TG3_TYPE_SCALAR || accessor.sparse.is_sparse)
    {
        return false;
    }

    int32_t stride = 0;
    const uint8_t* ptr =
        getAccessorElementPtr(model, accessor, elementIndex, stride);
    if (ptr == nullptr)
    {
        return false;
    }

    if (accessor.component_type == TG3_COMPONENT_TYPE_UNSIGNED_BYTE)
    {
        out = *ptr;
        return true;
    }
    if (accessor.component_type == TG3_COMPONENT_TYPE_UNSIGNED_SHORT)
    {
        uint16_t value;
        memcpy(&value, ptr, sizeof(value));
        out = value;
        return true;
    }
    if (accessor.component_type == TG3_COMPONENT_TYPE_UNSIGNED_INT)
    {
        uint32_t value;
        memcpy(&value, ptr, sizeof(value));
        out = value;
        return true;
    }

    return false;
}

static void parse_primitive_geometry(const tg3_model& model,
                                   const tg3_primitive& primitive,
                                   const glm::mat4& worldTransform,
                                   int materialOffset,
                                   std::vector<Triangle>& triangles)
{
    int32_t mode = primitive.mode == -1 ? TG3_MODE_TRIANGLES : primitive.mode;
    if (mode != TG3_MODE_TRIANGLES)
    {
        cerr << "Skipping non-triangle glTF primitive mode " << mode << endl;
        return;
    }

    int32_t positionAccessor = findAttribute(primitive, "POSITION");
    if (positionAccessor < 0 ||
        positionAccessor >= (int32_t)model.accessors_count)
    {
        cerr << "Skipping glTF primitive with no POSITION attribute" << endl;
        return;
    }

    int32_t normalAccessor = findAttribute(primitive, "NORMAL");
    int32_t materialId =
        primitive.material >= 0 && primitive.material < (int32_t)model.materials_count
            ? materialOffset + primitive.material
            : materialOffset;

    const tg3_accessor& positionData = model.accessors[positionAccessor];
    if (primitive.indices >= (int32_t)model.accessors_count)
    {
        cerr << "Skipping glTF primitive with invalid index accessor" << endl;
        return;
    }

    uint64_t indexCount = primitive.indices >= 0
                              ? model.accessors[primitive.indices].count
                              : positionData.count;
    if (indexCount < 3)
    {
        return;
    }

    glm::mat3 normalTransform =
        glm::mat3(glm::inverseTranspose(worldTransform));

    for (uint64_t i = 0; i + 2 < indexCount; i += 3)
    {
        uint32_t vertexIndices[3] = {
            (uint32_t)i,
            (uint32_t)i + 1,
            (uint32_t)i + 2,
        };

        if (primitive.indices >= 0)
        {
            if (!readAccessorIndex(model, primitive.indices, i,
                                   vertexIndices[0]) ||
                !readAccessorIndex(model, primitive.indices, i + 1,
                                   vertexIndices[1]) ||
                !readAccessorIndex(model, primitive.indices, i + 2,
                                   vertexIndices[2]))
            {
                cerr << "Skipping glTF primitive with unsupported indices"
                     << endl;
                return;
            }
        }

        glm::vec3 positions[3];
        if (!readAccessorVec3(model, positionAccessor, vertexIndices[0],
                              positions[0]) ||
            !readAccessorVec3(model, positionAccessor, vertexIndices[1],
                              positions[1]) ||
            !readAccessorVec3(model, positionAccessor, vertexIndices[2],
                              positions[2]))
        {
            cerr << "Skipping glTF primitive with unsupported POSITION data"
                 << endl;
            return;
        }

        Triangle triangle{};
        triangle.v0 = glm::vec3(worldTransform * glm::vec4(positions[0], 1.0f));
        triangle.v1 = glm::vec3(worldTransform * glm::vec4(positions[1], 1.0f));
        triangle.v2 = glm::vec3(worldTransform * glm::vec4(positions[2], 1.0f));

        if (normalAccessor >= 0)
        {
            glm::vec3 normals[3];
            if (readAccessorVec3(model, normalAccessor, vertexIndices[0],
                                 normals[0]) &&
                readAccessorVec3(model, normalAccessor, vertexIndices[1],
                                 normals[1]) &&
                readAccessorVec3(model, normalAccessor, vertexIndices[2],
                                 normals[2]))
            {
                triangle.n0 = glm::normalize(normalTransform * normals[0]);
                triangle.n1 = glm::normalize(normalTransform * normals[1]);
                triangle.n2 = glm::normalize(normalTransform * normals[2]);
            }
        }

        if (glm::dot(triangle.n0, triangle.n0) == 0.0f ||
            glm::dot(triangle.n1, triangle.n1) == 0.0f ||
            glm::dot(triangle.n2, triangle.n2) == 0.0f)
        {
            glm::vec3 faceNormal = glm::normalize(
                glm::cross(triangle.v1 - triangle.v0,
                           triangle.v2 - triangle.v0));
            triangle.n0 = faceNormal;
            triangle.n1 = faceNormal;
            triangle.n2 = faceNormal;
        }

        triangle.materialid = materialId;
        triangles.emplace_back(triangle);
    }
}

static void parseNodeGeometry(const tg3_model& model, int32_t nodeIndex,
                              const glm::mat4& parentTransform,
                              int materialOffset,
                              std::vector<Triangle>& triangles)
{
    if (nodeIndex < 0 || nodeIndex >= (int32_t)model.nodes_count)
    {
        return;
    }

    const tg3_node& node = model.nodes[nodeIndex];
    glm::mat4 worldTransform = parentTransform * getNodeLocalTransform(node);

    if (node.mesh >= 0 && node.mesh < (int32_t)model.meshes_count)
    {
        const tg3_mesh& mesh = model.meshes[node.mesh];
        for (uint32_t i = 0; i < mesh.primitives_count; ++i)
        {
            parse_primitive_geometry(model, mesh.primitives[i], worldTransform,
                                   materialOffset, triangles);
        }
    }

    for (uint32_t i = 0; i < node.children_count; ++i)
    {
        parseNodeGeometry(model, node.children[i], worldTransform,
                          materialOffset, triangles);
    }
}

static void parseSceneGeometry(const tg3_model& model,
                               const glm::mat4& rootTransform,
                               int materialOffset,
                               std::vector<Triangle>& triangles)
{
    int32_t sceneIndex = model.default_scene >= 0 ? model.default_scene : 0;

    if (sceneIndex >= 0 && sceneIndex < (int32_t)model.scenes_count)
    {
        const tg3_scene& scene = model.scenes[sceneIndex];
        for (uint32_t i = 0; i < scene.nodes_count; ++i)
        {
            parseNodeGeometry(model, scene.nodes[i], rootTransform,
                              materialOffset, triangles);
        }
        return;
    }

    for (uint32_t i = 0; i < model.nodes_count; ++i)
    {
        parseNodeGeometry(model, (int32_t)i, rootTransform, materialOffset,
                          triangles);
    }
}

static void setupDefaultGltfCamera(RenderState& state,
                                   const std::vector<Triangle>& triangles,
                                   const std::string& imageName)
{
    Camera& camera = state.camera;
    camera.resolution = glm::ivec2(800, 800);
    state.iterations = 5000;
    state.traceDepth = 8;
    state.imageName = imageName;
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f);

    glm::vec3 minPoint(0.0f);
    glm::vec3 maxPoint(0.0f);
    if (!triangles.empty())
    {
        minPoint = triangles[0].v0;
        maxPoint = triangles[0].v0;
        for (const Triangle& triangle : triangles)
        {
            minPoint = glm::min(minPoint, triangle.v0);
            minPoint = glm::min(minPoint, triangle.v1);
            minPoint = glm::min(minPoint, triangle.v2);
            maxPoint = glm::max(maxPoint, triangle.v0);
            maxPoint = glm::max(maxPoint, triangle.v1);
            maxPoint = glm::max(maxPoint, triangle.v2);
        }
    }

    glm::vec3 center = 0.5f * (minPoint + maxPoint);
    float radius = glm::length(maxPoint - minPoint) * 0.5f;
    if (radius <= 0.0f)
    {
        radius = 1.0f;
    }

    camera.lookAt = center;
    camera.position = center + glm::vec3(0.0f, 0.35f * radius, 2.5f * radius);
    finalizeCamera(state, 45.0f);
}

static void appendGltfFile(const std::string& gltfName,
                           const glm::mat4& rootTransform,
                           std::vector<Material>& materials,
                           std::vector<Triangle>& triangles) {
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

    int materialOffset = append_materials(model, materials);
    parseSceneGeometry(model, rootTransform, materialOffset, triangles);

    tg3_model_free(&model);
    tg3_error_stack_free(&errors);
}

void Scene::loadFromGltf(const std::string &gltfName) {
    materials.clear();
    triangles.clear();
    appendGltfFile(gltfName, glm::mat4(1.0f), materials, triangles);
    setupDefaultGltfCamera(state, triangles, gltfName);
}
