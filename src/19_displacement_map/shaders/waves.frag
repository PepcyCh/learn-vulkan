#version 450

layout(location = 0) in FragIn {
    vec3 pos;
    vec3 norm;
    vec2 texc;
    vec3 tan;
} fin;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform ObjectUniform {
    mat4 model;
    mat4 model_it;
    mat4 tex_transform;
    int mat_index;
} obj;

#define kTexturesCount 6
layout(set = 1, binding = 0) uniform sampler2D textures[kTexturesCount];

struct MaterialData {
    vec4 albedo;
    vec3 fresnel_r0;
    float roughness;
    mat4 mat_transform;
    int diffuse_index;
    int normal_index;
};
layout(set = 2, binding = 0) buffer readonly MaterialDataBuffer {
    MaterialData material_data[];
};

struct Light {
    vec3 strength;
    float falloff_start;
    vec3 direction;
    float falloff_end;
    vec3 position;
    float spot_power;
};

struct Material {
    vec4 albedo;
    vec3 fresnel_r0;
    float shininess;
};

#define kMaxLight 16
#define kDirLightCount 3
#define kPointLightCount 0
#define kSpotLightCount 0

layout(set = 3, binding = 1) uniform PassUniform {
    vec3 eye;
    float _0;
    float near;
    float far;
    float delta_time;
    float total_time;
    vec4 ambient;
    vec4 fog_color;
    float fog_start;
    float fog_end;
    vec2 _1;
    Light lights[kMaxLight];
} pass;

layout(set = 4, binding = 0) uniform samplerCube cubemap;

layout(set = 5, binding = 0) uniform WaveTexTransform {
    mat4 height_trans0;
    mat4 height_trans1;
    mat4 normal_trans0;
    mat4 normal_trans1;
} wave_tex_trans;

layout(set = 5, binding = 1) uniform sampler2D wave_tex0;
layout(set = 5, binding = 2) uniform sampler2D wave_tex1;

float CalcAttenuation(float d, float falloff_start, float falloff_end) {
    return clamp((falloff_end - d) / (falloff_end - falloff_start), 0.0f, 1.0f);
}

vec3 SchlickFresnel(vec3 r0, vec3 normal, vec3 light) {
    float cosine = max(dot(normal, light), 0.0f);
    float f0 = 1.0f - cosine;
    vec3 reflect_percent = r0 + (1.0f - r0) * (f0 * f0 * f0 * f0 * f0);
    return reflect_percent;
}

vec3 BlinnPhong(vec3 strength, vec3 light, vec3 normal, vec3 view, Material mat) {
    float m = mat.shininess * 256.0f;
    vec3 halfway = normalize(view + light);

    float roughness_factor = (m + 8.0f) * pow(max(dot(halfway, normal), 0.0f), m) / 8.0f;
    vec3 fresnel_factor = SchlickFresnel(mat.fresnel_r0, halfway, light);

    vec3 spec_albedo = fresnel_factor * roughness_factor;
    spec_albedo /= (spec_albedo + 1.0f);

    return (mat.albedo.rgb + spec_albedo) * strength;
}

vec3 ComputeDirectionalLight(Light L, Material mat, vec3 pos, vec3 normal, vec3 view) {
    vec3 light = -L.direction;
    float ndotl = max(dot(normal, light), 0.0f);
    vec3 strength = L.strength * ndotl;
    return BlinnPhong(strength, light, normal, view, mat);
}

vec3 ComputePointLight(Light L, Material mat, vec3 pos, vec3 normal, vec3 view) {
    vec3 light = L.position - pos;
    float d = length(light);
    if (d > L.falloff_end) {
        return vec3(0.0f, 0.0f, 0.0f);
    }
    light /= d;
    float ndotl = max(dot(normal, light), 0.0f);
    vec3 strength = L.strength * ndotl * CalcAttenuation(d, L.falloff_start, L.falloff_end);
    return BlinnPhong(strength, light, normal, view, mat);
}

vec3 ComputeSpotLight(Light L, Material mat, vec3 pos, vec3 normal, vec3 view) {
    vec3 light = L.position - pos;
    float d = length(light);
    if (d > L.falloff_end) {
        return vec3(0.0f, 0.0f, 0.0f);
    }
    light /= d;
    float ndotl = max(dot(normal, light), 0.0f);
    vec3 strength = L.strength * ndotl * CalcAttenuation(d, L.falloff_start, L.falloff_end);
    float spot_factor = pow(max(dot(-light, L.direction), 0.0f), L.spot_power);
    strength *= spot_factor;
    return BlinnPhong(strength, light, normal, view, mat);
}

vec4 ComputeLight(Light lights[kMaxLight], Material mat, vec3 pos, vec3 normal, vec3 view) {
    vec3 res = vec3(0.0f, 0.0f, 0.0f);

    int i = 0;
#if (kDirLightCount > 0)
    for (i = 0; i < kDirLightCount; i++) {
        res += ComputeDirectionalLight(lights[i], mat, pos, normal, view);
    }
#endif
#if (kPointLightCount > 0)
    for (i = kDirLightCount; i < kDirLightCount + kPointLightCount; i++) {
        res += ComputePointLight(lights[i], mat, pos, normal, view);
    }
#endif
#if (kSpotLightCount > 0)
    for (i = kDirLightCount + kPointLightCount; i < kDirLightCount + kPointLightCount + kSpotLightCount; i++) {
        res += ComputeSpotLight(lights[i], mat, pos, normal, view);
    }
#endif

    return vec4(res, 0.0f);
}

vec3 CalcBumpedNormal(vec3 norm_map, vec3 norm, vec3 tan) {
    // vec3 norm_z = 2.0f * norm_map - 1.0f;
    vec3 norm_z = norm_map;

    vec3 N = norm;
    vec3 T = normalize(tan - dot(tan, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);

    return TBN * norm_z;
}

void main() {
    vec2 texc = vec2(material_data[obj.mat_index].mat_transform * vec4(fin.texc, 0.0f, 1.0f));
    vec4 diffuse_tex = texture(textures[material_data[obj.mat_index].diffuse_index], texc);
#ifdef kAlphaTest
    if (diffuse_tex.a < 0.1f) {
        discard;
    }
#endif
    vec2 normal_uv0 = vec2(wave_tex_trans.normal_trans0 * vec4(fin.texc, 0.0f, 1.0f));
    vec2 normal_uv1 = vec2(wave_tex_trans.normal_trans1 * vec4(fin.texc, 0.0f, 1.0f));
    vec3 norm_map = normalize(texture(wave_tex0, normal_uv0).xyz * 2.0f +
        texture(wave_tex1, normal_uv1).xyz * 2.0f - 2.0f);
    vec3 norm = normalize(fin.norm);
    norm = CalcBumpedNormal(norm_map, norm, fin.tan);
    vec3 view = pass.eye - fin.pos;
    float view_dist = length(view);
    view = normalize(view);
    vec4 ambient = pass.ambient * material_data[obj.mat_index].albedo;
    float shininess = (1.0f - material_data[obj.mat_index].roughness);
    vec4 diffuse = material_data[obj.mat_index].albedo * diffuse_tex;
    Material mat = { diffuse, material_data[obj.mat_index].fresnel_r0, shininess };
    vec4 light_res = ComputeLight(pass.lights, mat, fin.pos, norm, view);
    vec4 res = light_res + ambient;

    vec3 r = reflect(-view, norm);
    vec3 reflected_color = texture(cubemap, r).rgb;
    vec3 fresnel_factor = SchlickFresnel(mat.fresnel_r0, norm, r);
    res.rgb += shininess * fresnel_factor * reflected_color;

    res.a = diffuse.a;
    out_color = res;
}