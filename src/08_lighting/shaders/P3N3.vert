#version 450

layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec3 vin_norm;

layout(location = 0) out struct VertexOut {
    vec3 pos;
    vec3 norm;
} vout;

layout(set = 0, binding = 0) uniform ObjectUniform {
    mat4 model;
    mat4 model_it;
} obj;

layout(set = 2, binding = 0) uniform PassUniform {
    mat4 proj;
    mat4 view;
} pass;

void main() {
    vec4 pos_w = obj.model * vec4(vin_pos, 1.0);
    vout.pos = vec3(pos_w);
    vout.norm = vec3(obj.model_it * vec4(vin_norm, 0.0f));
    gl_Position = pass.proj * pass.view * pos_w;
}
