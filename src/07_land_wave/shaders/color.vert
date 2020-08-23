#version 450

layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec3 vin_color;

layout(location = 0) out vec3 color;

layout(set = 0, binding = 0) uniform ObjectUniform {
    mat4 model;
} obj_ub;

layout(set = 1, binding = 0) uniform PassUniform {
    mat4 proj;
    mat4 view;
} pass_ub;

void main() {
    gl_Position = pass_ub.proj * pass_ub.view * obj_ub.model * vec4(vin_pos, 1.0);
    color = vin_color;
}
