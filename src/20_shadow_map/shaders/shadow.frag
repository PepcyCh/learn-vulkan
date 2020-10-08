#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in FragIn {
    vec2 texc;
} fin;

layout(set = 0, binding = 0) uniform ObjectUniform {
    mat4 model;
    mat4 model_it;
    mat4 tex_transform;
    int mat_index;
} obj;

layout(set = 1, binding = 0) uniform sampler2D textures[];

struct MaterialData {
    vec4 albedo;
    vec3 fresnel_r0;
    float roughness;
    mat4 mat_transform;
    int diffuse_index;
    int normal_index;
};
layout(set = 2, binding = 0) buffer readonly MaterialDataBuffer {
    MaterialData material_data[];
};

void main() {
#ifdef kAlphaTest
    vec2 texc = vec2(material_data[obj.mat_index].mat_transform * vec4(fin.texc, 0.0f, 1.0f));
    vec4 diffuse_tex = texture(textures[material_data[obj.mat_index].diffuse_index], texc);
    if (diffuse_tex.a < 0.1f) {
        discard;
    }
#endif
}