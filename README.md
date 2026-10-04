CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.


- Added ideal diffuse with actual path tracing
- added anti aliasing
- added stream comopaction + material based optimization
- Added reflective and alpha materials
- added gltf loading
- added BVH Tree for mesh acceleration
- added gltf textures
- Added metalics (reflective materials)
- Added microfacets (rough vs smooth metals)
- Added dielectrics (rough vs smooth, both with )
- Added mixed opaque materials (partially dielectric + opaque + rough + metallic)
- Added OIDN GPU Denoiser with beauty, albedo, and normal buffers.

- Ideal diffuse path tracing with anti-aliasing
- Stream compaction and material sorting optimization
- Reflective, metallic, alpha, and emissive materials
- glTF mesh loading with transforms, materials, textures, normal maps, and sRGB decode
- BVH acceleration for triangle meshes
- Microfacet GGX rough/smooth metals
- Smooth and rough dielectric glass with IOR/transmission
- Mixed diffuse/metallic/transmissive materials
- Next-event estimation, shadow rays, and MIS for emissive geometry
- Procedural environment lighting toggle with sky/sun/ground color
- OIDN GPU denoiser using beauty, albedo, and normal buffers