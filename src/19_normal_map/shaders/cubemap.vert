#version 450

layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec3 vin_norm;
layout(location = 2) in vec2 vin_texc;
layout(location = 3) in vec3 vin_tan;

layout(location = 0) out VertexOut {
    vec3 dir;
} vout;

layout(set = 0, binding = 0) uniform ObjectUniform {
    mat4 model;
    mat4 model_it;
    mat4 tex_transform;
} obj;

layout(set = 3, binding = 0) uniform PassUniform0 {
    mat4 proj;
    mat4 view;
} pass0;

layout(set = 3, binding = 1) uniform PassUniform1 {
    vec3 eye;
    float _0;
} pass1;

void main() {
    vout.dir = vin_pos;

    vec4 pos_w = obj.model * vec4(vin_pos, 1.0);
    pos_w.xyz += pass1.eye;
    gl_Position = (pass0.proj * pass0.view * pos_w).xyww;
}
