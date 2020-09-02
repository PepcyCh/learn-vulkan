#version 450

layout(vertices = 4) out;

layout(set = 0, binding = 0) uniform ObjectUniform {
    mat4 model;
    mat4 model_it;
    mat4 tex_transform;
} obj;

layout(set = 3, binding = 1) uniform PassUniform {
    vec3 eye;
    float _0;
    float near;
    float far;
    float delta_time;
    float total_time;
} pass;

void main() {
    vec4 center = 0.25f * (gl_in[0].gl_Position + gl_in[1].gl_Position + gl_in[2].gl_Position + gl_in[3].gl_Position);
    center = obj.model * center;
    float dist = length(center.xyz - pass.eye);

    float d0 = 20.0f;
    float d1 = 100.0f;
    float tess = 63.0f * clamp((d1 - dist) / (d1 - d0), 0.0f, 1.0f) + 1.0f;

    gl_TessLevelOuter[0] = tess;
    gl_TessLevelOuter[1] = tess;
    gl_TessLevelOuter[2] = tess;
    gl_TessLevelOuter[3] = tess;
    gl_TessLevelInner[0] = tess;
    gl_TessLevelInner[1] = tess;

    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
