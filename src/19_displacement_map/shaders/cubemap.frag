#version 450

layout(location = 0) in FragIn {
    vec3 dir;
} fin;

layout(location = 0) out vec4 out_color;

layout(set = 4, binding = 0) uniform samplerCube cubemap;

void main() {
    out_color = texture(cubemap, fin.dir);
}
