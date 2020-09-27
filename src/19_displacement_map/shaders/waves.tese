#version 450

layout(quads, equal_spacing, cw) in;

layout(location = 0) in TeseIn {
    vec3 norm;
    vec2 texc;
    vec3 tan;
} tein[];

layout(location = 0) out TeseOut {
    vec3 pos;
    vec3 norm;
    vec2 texc;
    vec3 tan;
} teout;

layout(set = 0, binding = 0) uniform ObjectUniform {
    mat4 model;
    mat4 model_it;
    mat4 tex_transform;
} obj;

layout(set = 3, binding = 0) uniform PassUniform {
    mat4 proj;
    mat4 view;
} pass;

layout(set = 5, binding = 0) uniform WaveTexTransform {
    mat4 height_trans0;
    mat4 height_trans1;
    mat4 normal_trans0;
    mat4 normal_trans1;
    vec2 height_scale;
} wave_tex_trans;

layout(set = 5, binding = 1) uniform sampler2D wave_tex0;
layout(set = 5, binding = 2) uniform sampler2D wave_tex1;

void main() {
    vec2 uv = gl_TessCoord.xy;

    vec3 p0 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, uv.x).xyz;
    vec3 p1 = mix(gl_in[2].gl_Position, gl_in[3].gl_Position, uv.x).xyz;
    vec3 p = mix(p0, p1, uv.y);

    vec3 norm0 = mix(tein[0].norm, tein[1].norm, uv.x);
    vec3 norm1 = mix(tein[2].norm, tein[3].norm, uv.x);
    vec3 norm = normalize(mix(norm0, norm1, uv.y));

    vec2 texc0 = mix(tein[0].texc, tein[1].texc, uv.x);
    vec2 texc1 = mix(tein[2].texc, tein[3].texc, uv.x);
    vec2 texc = mix(texc0, texc1, uv.y);

    vec2 height_uv0 = vec2(wave_tex_trans.height_trans0 * vec4(texc, 0.0f, 1.0f));
    vec2 height_uv1 = vec2(wave_tex_trans.height_trans1 * vec4(texc, 0.0f, 1.0f));
    float height = wave_tex_trans.height_scale.x * texture(wave_tex0, height_uv0).w +
        wave_tex_trans.height_scale.y * texture(wave_tex1, height_uv1).w;
    p.y += height;

    vec3 tan0 = mix(tein[0].tan, tein[1].tan, uv.x);
    vec3 tan1 = mix(tein[2].tan, tein[3].tan, uv.x);
    vec3 tan = normalize(mix(tan0, tan1, uv.y));

    vec4 pos_w = obj.model * vec4(p, 1.0f);
    gl_Position = pass.proj * pass.view * pos_w;
    teout.pos = vec3(pos_w);
    teout.norm = vec3(obj.model_it * vec4(norm, 1.0f));
    teout.texc = texc;
    teout.tan = vec3(obj.model_it * vec4(tan, 1.0f));
}
