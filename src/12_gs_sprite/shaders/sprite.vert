#version 450

layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec2 vin_size;

layout(location = 0) out VertexOut {
    vec2 size;
} vout;

void main() {
    gl_Position = vec4(vin_pos, 1.0f);
    vout.size = vin_size;
}
