#include "MathUtil.h"

float MathUtil::Radians(float degree) {
    return degree / 180.0f * kPi;
}
float MathUtil::Degree(float radians) {
    return radians / kPi * 180.0f;
}
float MathUtil::Lerp(float a, float b, float t) {
    return a + t * (b - a);
}

Eigen::Vector3f MathUtil::TransformPoint(const Eigen::Matrix4f &mat, const Eigen::Vector3f &point) {
    auto temp = mat * Eigen::Vector4f(point.x(), point.y(), point.z(), 1.0f);
    float inv_w = 1.0f / temp.w();
    return temp.head<3>() * inv_w;
}
Eigen::Vector3f MathUtil::TransformVector(const Eigen::Matrix4f &mat, const Eigen::Vector3f &vec) {
    return (mat * Eigen::Vector4f(vec.x(), vec.y(), vec.z(), 0.0f)).head<3>();
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
Eigen::Matrix4f MathUtil::Reflect(const Eigen::Vector4f &plane) {
    float mod = Eigen::Vector3f(plane.x(), plane.y(), plane.z()).norm();
    float inv_mod = 1.0f / mod;
    Eigen::Vector4f normalized_plane = plane * inv_mod;

    // A^2 + B^2 + C^2 = 1
    // 1 - 2A^2    -2AB        -2AC        -2AD
    // -2BA        1 - 2B^2    -2BC        -2BD
    // -2CA        -2CB        1 - 2C^2    -2CD
    // 0           0           0           1
    Eigen::Matrix4f res = Eigen::Matrix4f::Identity() - 2.0f * normalized_plane * normalized_plane.transpose();
    res.row(3) = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
    return res;
}

Eigen::Matrix4f MathUtil::LookAt(const Eigen::Vector3f &pos, const Eigen::Vector3f &look_at,
    const Eigen::Vector3f &up) {
    Eigen::Vector3f w = (pos - look_at).normalized();
    Eigen::Vector3f u = up.cross(w).normalized();
    Eigen::Vector3f v = w.cross(u);
    return ViewTransform(pos, u, v, w);
}
Eigen::Matrix4f MathUtil::ViewTransform(const Eigen::Vector3f &pos, const Eigen::Vector3f &u,
    const Eigen::Vector3f &v, const Eigen::Vector3f &w) {
    Eigen::Matrix4f m;
    m.row(0) = Eigen::Vector4f(u.x(), u.y(), u.z(), -u.dot(pos));
    m.row(1) = Eigen::Vector4f(v.x(), v.y(), v.z(), -v.dot(pos));
    m.row(2) = Eigen::Vector4f(w.x(), w.y(), w.z(), -w.dot(pos));
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

int MathUtil::RandI(int l, int r) {
    if (!rnd_init) {
        rnd_gen.seed(rnd_dv());
        rnd_init = true;
    }
    std::uniform_int_distribution<> rnd_int(l, r);
    return rnd_int(rnd_gen);
}
float MathUtil::RandF(float l, float r) {
    if (!rnd_init) {
        rnd_gen.seed(rnd_dv());
        rnd_init = true;
    }
    std::uniform_real_distribution<> rnd_real(l, r);
    return rnd_real(rnd_gen);
}