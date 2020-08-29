#version 450

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in GeometryIn {
    vec2 size;
} gin[];

layout(location = 0) out GeometryOut {
    vec3 pos;
    vec3 norm;
    vec2 texc;
    flat int prim_id;
} gout;

layout(set = 3, binding = 0) uniform PassUniform0 {
    mat4 proj;
    mat4 view;
} pass0;

layout(set = 3, binding = 1) uniform PassUniform1 {
    vec3 eye;
} pass1;

void main() {
    vec3 up = vec3(0.0f, 1.0f, 0.0f);
    vec3 look = pass1.eye - gl_in[0].gl_Position.xyz;
    look.y = 0.0f;
    look = normalize(look);
    vec3 right = cross(look, up);

    float hw = 0.5f * gin[0].size.x;
    float hh = 0.5f * gin[0].size.y;

    vec4 pos[4] = {
        vec4(gl_in[0].gl_Position.xyz + hw * right + hh * up, 1.0f),
        vec4(gl_in[0].gl_Position.xyz + hw * right - hh * up, 1.0f),
        vec4(gl_in[0].gl_Position.xyz - hw * right + hh * up, 1.0f),
        vec4(gl_in[0].gl_Position.xyz - hw * right - hh * up, 1.0f)
    };

    vec2 texc[4] = {
        vec2(0.0f, 0.0f),
        vec2(0.0f, 1.0f),
        vec2(1.0f, 0.0f),
        vec2(1.0f, 1.0f)
    };

    for (int i = 0; i < 4; i++) {
        gout.pos = vec3(pos[i]);
        gl_Position = pass0.proj * pass0.view * pos[i];
        gout.norm = look;
        gout.texc = texc[i];
        gout.prim_id = gl_PrimitiveIDIn;
        EmitVertex();
    }
    EndPrimitive();
}
