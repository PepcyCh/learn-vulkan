#pragma once

#include <random>

#include "Eigen/Dense"

class MathUtil {
public:
    inline static const float PI = 3.141592653589793238463f;
    inline static const float PI_INV = 0.3183098861837907f;
    inline static const float PI2 = 2.0f * PI;
    inline static const float PI_DIV_2 = 0.5f * PI;
    inline static const float PI_DIV_4 = 0.25f * PI;

    static float Radians(float degree);
    static float Degree(float radians);
    static float Lerp(float a, float b, float t);

    static Eigen::Matrix4f AngelAxis(float angel, const Eigen::Vector3f &axis);
    static Eigen::Matrix4f Scale(const Eigen::Vector3f &scale);
    static Eigen::Matrix4f Translate(const Eigen::Vector3f &trans);

    static Eigen::Matrix4f LookAt(const Eigen::Vector3f &eye, const Eigen::Vector3f &look_at,
                                  const Eigen::Vector3f &up);
    static Eigen::Matrix4f Perspective(float fov, float aspect, float near, float far, bool flip_y = false);

    static int RandI(int l, int r);
    static float RandF(float l, float r);

private:
    inline static bool rnd_init = false;
    inline static std::random_device rnd_dv = std::random_device();
    inline static std::mt19937 rnd_gen = std::mt19937();
};