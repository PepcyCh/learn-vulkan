#pragma once

#include "Eigen/Dense"

class MathUtil {
public:
    inline static const float PI = 3.141592653589793238463f;
    inline static const float PI_INV = 0.3183098861837907f;

    static float Radians(float degree);
    static float Degree(float radians);
    static float Lerp(float a, float b, float t);

    static Eigen::Matrix4f AngelAxis(float angel, const Eigen::Vector3f &axis);
    static Eigen::Matrix4f Scale(const Eigen::Vector3f &scale);
    static Eigen::Matrix4f Translate(const Eigen::Vector3f &trans);

    static Eigen::Matrix4f LookAt(const Eigen::Vector3f &eye, const Eigen::Vector3f &look_at,
                                  const Eigen::Vector3f &up);
    static Eigen::Matrix4f Perspective(float fov, float aspect, float near, float far, bool flip_y = false);
};