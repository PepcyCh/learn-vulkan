#version 450

layout(location = 0) in vec3 vin_pos;

void main() {
    gl_Position = vec4(vin_pos, 1.0f);
}
