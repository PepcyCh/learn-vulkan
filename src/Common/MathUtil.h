#pragma once

#include <random>

#include "Eigen/Dense"

enum class Containment {
    eDisjoint,
    eIntersects,
    eContains,
};

struct BoundingBox {
    Eigen::Vector3f center = { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f extent = { 0.0f, 0.0f, 0.0f };
};

struct Frustum {
    Frustum() = default;
    explicit Frustum(const Eigen::Matrix4f &proj, bool proj_flip_y = false);

    Containment Contains(const BoundingBox &bbox) const;

    Eigen::Vector3f origin;
    Eigen::Quaternionf orientation;
    float right_slope;
    float left_slope;
    float top_slope;
    float bottom_slope;
    float near;
    float far;
};

struct Ray {
    bool IntersectWithBbox(const BoundingBox &bbox, float &tmin);
    bool IntersectWithTriangle(const Eigen::Vector3f &v0, const Eigen::Vector3f &v1, const Eigen::Vector3f &v2,
        float &tmin);

    Eigen::Vector3f origin;
    Eigen::Vector3f dir;
};

class MathUtil {
public:
    inline static const float kPi = 3.141592653589793238463f;
    inline static const float kPiInv = 0.3183098861837907f;
    inline static const float k2Pi = 2.0f * kPi;
    inline static const float kPiDiv2 = 0.5f * kPi;
    inline static const float kPiDiv4 = 0.25f * kPi;

    static float Radians(float degree);
    static float Degree(float radians);
    static float Lerp(float a, float b, float t);

    static Eigen::Vector3f TransformPoint(const Eigen::Matrix4f &mat, const Eigen::Vector3f &point);
    static Eigen::Vector3f TransformVector(const Eigen::Matrix4f &mat, const Eigen::Vector3f &vec);
    static BoundingBox TransformBoundingBox(const Eigen::Matrix4f &mat, const BoundingBox &bbox);
    static Frustum TransformFrustum(const Eigen::Matrix4f &mat, const Frustum &frustum);
    static Ray TransformRay(const Eigen::Matrix4f &mat, const Ray &ray);

    static Eigen::Matrix4f AngelAxis(float angel, const Eigen::Vector3f &axis);
    static Eigen::Matrix4f Scale(const Eigen::Vector3f &scale);
    static Eigen::Matrix4f Translate(const Eigen::Vector3f &trans);
    static Eigen::Matrix4f Reflect(const Eigen::Vector4f &plane);

    static Eigen::Matrix4f LookAt(const Eigen::Vector3f &pos, const Eigen::Vector3f &look_at,
        const Eigen::Vector3f &up);
    static Eigen::Matrix4f ViewTransform(const Eigen::Vector3f &pos, const Eigen::Vector3f &right,
        const Eigen::Vector3f &up, const Eigen::Vector3f &back);
    static Eigen::Matrix4f Perspective(float fov, float aspect, float near, float far, bool flip_y = false);
    static Eigen::Matrix4f Orthographic(float l, float r, float b, float t, float n, float f);

    static int RandI(int l, int r);
    static float RandF(float l, float r);

    static Eigen::Vector3f Spherical2Cartesian(float radius, float theta, float phi) {
        return {
            radius * std::sin(phi) * std::cos(theta),
            radius * std::cos(phi),
            radius * std::sin(phi) * std::sin(theta)
        };
    }

private:
    inline static bool rnd_init = false;
    inline static std::random_device rnd_dv = std::random_device();
    inline static std::mt19937 rnd_gen = std::mt19937();
};