#version 450

layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec3 vin_norm;
layout(location = 2) in vec2 vin_texc;

layout(location = 0) out VertexOut {
    vec3 pos;
    vec3 norm;
    vec2 texc;
} vout;

layout(set = 0, binding = 0) uniform ObjectUniform {
    mat4 model;
    mat4 model_it;
    mat4 tex_transform;
    vec2 displacement_map_texel_size;
    float grid_spatial_step;
} obj;

layout(set = 3, binding = 0) uniform PassUniform {
    mat4 proj;
    mat4 view;
} pass;

layout(set = 4, binding = 0) uniform sampler2D displacement_map;

void main() {
    vec3 pos = vin_pos;
    vec3 norm = vin_norm;
#ifdef kUseDisplacementMap
    pos.y += texture(displacement_map, vin_texc).x;
    float du = obj.displacement_map_texel_size.x;
    float dv = obj.displacement_map_texel_size.y;
    float l = texture(displacement_map, vin_texc - vec2(du, 0.0f)).x;
    float r = texture(displacement_map, vin_texc + vec2(du, 0.0f)).x;
    float t = texture(displacement_map, vin_texc - vec2(0.0f, dv)).x;
    float b = texture(displacement_map, vin_texc + vec2(0.0f, dv)).x;
    norm = normalize(vec3(l - r, 2.0f * obj.grid_spatial_step, b - t));
#endif
    vec4 pos_w = obj.model * vec4(pos, 1.0);
    vout.pos = vec3(pos_w);
    vout.norm = vec3(obj.model_it * vec4(norm, 0.0f));
    vout.texc = vec2(obj.tex_transform * vec4(vin_texc, 0.0f, 1.0f));
    gl_Position = pass.proj * pass.view * pos_w;
}
