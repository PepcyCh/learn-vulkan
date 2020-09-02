#version 450

layout(quads, equal_spacing, cw) in;
// layout(quads, fractional_even_spacing, cw) in;
// layout(quads, fractional_odd_spacing, cw) in;

layout(location = 0) out TessOut {
    vec3 pos;
    vec3 norm;
    vec2 texc;
} tout;

layout(set = 0, binding = 0) uniform ObjectUniform {
    mat4 model;
    mat4 model_it;
    mat4 tex_transform;
} obj;

layout(set = 3, binding = 0) uniform PassUniform {
    mat4 proj;
    mat4 view;
} pass;

void main() {
    vec3 p0 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x).xyz;
    vec3 p1 = mix(gl_in[2].gl_Position, gl_in[3].gl_Position, gl_TessCoord.x).xyz;
    vec3 p = mix(p0, p1, gl_TessCoord.y);
    p.y = 0.3f * (p.z * sin(p.x) + p.x * cos(p.z));
    vec3 n = normalize(vec3(-0.3f * p.z * cos(p.x) - 0.3f * cos(p.z), 1.0f, -0.3f * sin(p.x) + 0.3f * p.x * sin(p.z)));

    vec4 posw = obj.model * vec4(p, 1.0f);
    gl_Position = pass.proj * pass.view * posw;
    tout.pos = vec3(posw);
    tout.norm = vec3(obj.model_it * vec4(n, 1.0f));
    tout.texc = vec2(obj.tex_transform * vec4(gl_TessCoord.xy, 0.0f, 1.0f));
}
