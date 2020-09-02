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

vec4 BernsteinBasic(float t) {
    float it = 1.0f - t;
    return vec4(it * it * it, 3.0f * it * it * t, 3.0f * it * t * t, t * t * t);
}

vec4 DBernsteinBasic(float t) {
    float it = 1.0f - t;
    return vec4(-3 * it * it, 3 * it * it - 6 * it * t, 6 * t * it - 3 * t * t, 3 * t * t);
}

vec3 CubizBezierSum(vec4 basic_u, vec4 basic_v) {
    vec3 sum = vec3(0.0f);
    sum = basic_v.x * (basic_u.x * gl_in[0].gl_Position + basic_u.y * gl_in[1].gl_Position +
        basic_u.z * gl_in[2].gl_Position + basic_u.w * gl_in[3].gl_Position).xyz;
    sum += basic_v.y * (basic_u.x * gl_in[4].gl_Position + basic_u.y * gl_in[5].gl_Position +
        basic_u.z * gl_in[6].gl_Position + basic_u.w * gl_in[7].gl_Position).xyz;
    sum += basic_v.z * (basic_u.x * gl_in[8].gl_Position + basic_u.y * gl_in[9].gl_Position +
        basic_u.z * gl_in[10].gl_Position + basic_u.w * gl_in[11].gl_Position).xyz;
    sum += basic_v.w * (basic_u.x * gl_in[12].gl_Position + basic_u.y * gl_in[13].gl_Position +
        basic_u.z * gl_in[14].gl_Position + basic_u.w * gl_in[15].gl_Position).xyz;
    return sum;
}

void main() {
    vec4 basic_u = BernsteinBasic(gl_TessCoord.x);
    vec4 basic_v = BernsteinBasic(gl_TessCoord.y);

    vec4 d_basic_u = DBernsteinBasic(gl_TessCoord.x);
    vec4 d_basic_v = DBernsteinBasic(gl_TessCoord.y);

    vec3 pos = CubizBezierSum(basic_u, basic_v);
    vec3 dpdu = CubizBezierSum(d_basic_u, basic_v);
    vec3 dpdv = CubizBezierSum(basic_u, d_basic_v);
    
    vec4 posw = obj.model * vec4(pos, 1.0f);
    tout.pos = vec3(posw);
    gl_Position = pass.proj * pass.view * posw;
    tout.norm = vec3(obj.model_it * vec4(cross(normalize(dpdu), normalize(dpdv)), 1.0f));
    tout.texc = vec2(obj.tex_transform * vec4(gl_TessCoord.xy, 0.0f, 1.0f));
}
