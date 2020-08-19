#version 450

layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec3 vin_color;

layout(location = 0) out vec3 color;

layout(binding = 0) uniform UniformBuffer {
    mat4 proj;
    mat4 view;
    mat4 model;
} ub;

void main() {
    gl_Position = ub.proj * ub.view * ub.model * vec4(vin_pos, 1.0);
    color = vin_color;
}
