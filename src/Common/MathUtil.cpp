#include "MathUtil.h"

float MathUtil::Radians(float degree) {
    return degree / 180.0f * PI;
}
float MathUtil::Degree(float radians) {
    return radians / PI * 180.0f;
}
float MathUtil::Lerp(float a, float b, float t) {
    return a + t * (b - a);
}

Eigen::Matrix4f MathUtil::AngelAxis(float angel, const Eigen::Vector3f &axis) {
    Eigen::Affine3f a;
    a = Eigen::AngleAxisf(angel, axis);
    return a.matrix();
}
Eigen::Matrix4f MathUtil::Scale(const Eigen::Vector3f &scale) {
    Eigen::Affine3f a;
    a = Eigen::Scaling(scale);
    return a.matrix();

}
Eigen::Matrix4f MathUtil::Translate(const Eigen::Vector3f &trans) {
    Eigen::Affine3f a;
    a = Eigen::Translation3f(trans);
    return a.matrix();
}

Eigen::Matrix4f MathUtil::LookAt(const Eigen::Vector3f &eye, const Eigen::Vector3f &look_at,
                                 const Eigen::Vector3f &up) {
    Eigen::Vector3f w = (eye - look_at).normalized();
    Eigen::Vector3f u = up.cross(w).normalized();
    Eigen::Vector3f v = w.cross(u);
    Eigen::Matrix4f m;
    m.row(0) = Eigen::Vector4f(u.x(), u.y(), u.z(), -u.dot(eye));
    m.row(1) = Eigen::Vector4f(v.x(), v.y(), v.z(), -v.dot(eye));
    m.row(2) = Eigen::Vector4f(w.x(), w.y(), w.z(), -w.dot(eye));
    m.row(3) = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
    return m;
}
Eigen::Matrix4f MathUtil::Perspective(float fov, float aspect, float near, float far, bool flip_y) {
    float t = 1.0f / std::tan(fov * 0.5f);
    float inv = 1.0f / (far - near);
    Eigen::Matrix4f m;
    m.setZero();
    m(0, 0) = t / aspect;
    m(1, 1) = t;
    m(2, 2) = -far * inv;
    m(2, 3) = -near * far * inv;
    m(3, 2) = -1.0f;
    if (flip_y) {
        m(1, 1) = -m(1, 1);
    }
    return m;
}