#version 450

layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec3 vin_norm;
layout(location = 2) in vec2 vin_texc;
layout(location = 3) in vec3 vin_tan;

layout(location = 0) out VertexOut {
    vec3 norm;
    vec2 texc;
    vec3 tan;
} vout;

void main() {
    vout.norm = vin_norm;
    vout.texc = vin_texc;
    vout.tan = vin_tan;
    gl_Position = vec4(vin_pos, 1.0f);
}
