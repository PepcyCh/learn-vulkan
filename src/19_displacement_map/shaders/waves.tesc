#version 450

layout(vertices = 4) out;

layout(location = 0) in TescIn {
    vec3 norm;
    vec2 texc;
    vec3 tan;
} tcin[];

layout(location = 0) out TescOut {
    vec3 norm;
    vec2 texc;
    vec3 tan;
} tcout[];

void main() {
    float tess = 64.0f;
    gl_TessLevelOuter[0] = tess;
    gl_TessLevelOuter[1] = tess;
    gl_TessLevelOuter[2] = tess;
    gl_TessLevelOuter[3] = tess;
    gl_TessLevelInner[0] = tess;
    gl_TessLevelInner[1] = tess;

    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    tcout[gl_InvocationID].norm = tcin[gl_InvocationID].norm;
    tcout[gl_InvocationID].texc = tcin[gl_InvocationID].texc;
    tcout[gl_InvocationID].tan = tcin[gl_InvocationID].tan;
}
