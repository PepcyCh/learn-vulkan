#version 450

layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec3 vin_norm;
layout(location = 2) in vec2 vin_texc;

layout(location = 0) out VertexOut {
    vec3 pos;
    vec3 norm;
    vec2 texc;
    flat int mat_index;
} vout;

struct InstanceData {
    mat4 model;
    mat4 model_it;
    mat4 tex_transform;
    int mat_index;
};
layout(set = 0, binding = 0) buffer InstanceDataBuffer {
    InstanceData instance_data[];
};

layout(set = 3, binding = 0) uniform PassUniform {
    mat4 proj;
    mat4 view;
} pass;

void main() {
    vec4 pos_w = instance_data[gl_InstanceIndex].model * vec4(vin_pos, 1.0);
    vout.pos = vec3(pos_w);
    vout.norm = vec3(instance_data[gl_InstanceIndex].model_it * vec4(vin_norm, 0.0f));
    vout.texc = vec2(instance_data[gl_InstanceIndex].tex_transform * vec4(vin_texc, 0.0f, 1.0f));
    gl_Position = pass.proj * pass.view * pos_w;
    vout.mat_index = instance_data[gl_InstanceIndex].mat_index;
}
