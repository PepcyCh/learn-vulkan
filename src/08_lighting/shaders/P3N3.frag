#version 450

layout(location = 0) in FragIn {
    vec3 pos;
    vec3 norm;
} fin;

layout(location = 0) out vec4 out_color;

layout(set = 1, binding = 0) uniform MaterialUniform {
    vec4 albedo;
    vec3 fresnel_r0;
    float roughness;
    mat4 mat_transform;
} mat;

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
#define kDirLightCount 1
#define kPointLightCount 0
#define kSpotLightCount 0

layout(set = 2, binding = 1) uniform PassUniform {
    vec3 eye;
    float _0;
    float near;
    float far;
    float delta_time;
    float total_time;
    vec4 ambient;
    Light lights[kMaxLight];
} pass;

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

void main() {
    vec3 norm = normalize(fin.norm);
    vec3 view = normalize(pass.eye - fin.pos);
    vec4 ambient = pass.ambient * mat.albedo;
    float shininess = 1.0f - mat.roughness;
    Material mat = { mat.albedo, mat.fresnel_r0, shininess };
    vec4 light_res = ComputeLight(pass.lights, mat, fin.pos, norm, view);
    vec4 res = light_res + ambient;
    res.a = mat.albedo.a;
    out_color = res;
}
