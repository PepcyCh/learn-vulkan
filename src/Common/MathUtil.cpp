#include "MathUtil.h"

#include <limits>

namespace {

bool DoesRayIntersectSegment(const Ray &r, const Eigen::Vector3f &v0,
    const Eigen::Vector3f &v1, float &t) {
    Eigen::Vector3f u = v1 - v0;
    Eigen::Vector3f v = u.cross(r.dir);
    if (v.norm() == 0) {
        return false;
    }
    Eigen::Vector3f w = u.cross(v);
    float c = w.dot(v0);
    float temp = w.dot(r.dir);
    if (temp == 0) {
        return false;
    }
    t = (c - w.dot(r.origin)) / temp;
    Eigen::Vector3f p = r.origin + r.dir * t;
    float ut = (p - v0).dot(u) / u.norm();
    return ut >= 0.0f && ut <= 1.0f;
}

}

Frustum::Frustum(const Eigen::Matrix4f &proj, bool proj_flip_y) {
    origin = { 0.0f, 0.0f, 0.0f };
    // ctor of Eigen::Quaternion is (w, x, y, z) instead of (x, y, z, w)
    orientation = { 1.0f, 0.0f, 0.0f, 0.0f };

    Eigen::Vector4f points[6] = {
        { 1.0f, 0.0f, 1.0f, 1.0f },
        { -1.0f, 0.0f, 1.0f, 1.0f },
        { 0.0f, 1.0f, 1.0f, 1.0f },
        { 0.0f, -1.0f, 1.0f, 1.0f },
        { 0.0f, 0.0f, 0.0f, 1.0f },
        { 0.0f, 0.0f, 1.0f, 1.0f }
    };
    Eigen::Matrix4f proj_inv = proj.inverse();
    for (auto &point : points) {
        point = proj_inv * point;
    }

    right_slope = points[0].x() / points[0].z(); // neg
    left_slope = points[1].x() / points[1].z(); // pos
    top_slope = points[2].y() / points[2].z(); // neg
    bottom_slope = points[3].y() / points[3].z(); // pos
    near = points[4].z() / points[4].w(); // neg
    far = points[5].z() / points[5].w(); // neg
    if (proj_flip_y) {
        top_slope = -top_slope;
        bottom_slope = -bottom_slope;
    }
}

Containment Frustum::Contains(const BoundingBox &bbox) const {
    Eigen::Vector3f bmax = bbox.center + bbox.extent - origin;
    Eigen::Vector3f bmin = bbox.center - bbox.extent - origin;

    Eigen::Vector3f plane_normal[6] = {
        { -1.0f, 0.0f, right_slope },
        { 1.0f, 0.0f, -left_slope },
        { 0.0f, -1.0f, top_slope },
        { 0.0f, 1.0f, -bottom_slope },
        { 0.0f, 0.0f, -1.0f }, // near
        { 0.0f, 0.0f, 1.0f }, // far
    };
    for (int i = 0; i < 6; i++) {
        Eigen::Quaternion qtemp(0.0f, plane_normal[i].x(), plane_normal[i].y(), plane_normal[i].z());
        qtemp = orientation * qtemp * orientation.conjugate();
        plane_normal[i] = qtemp.vec();

        Eigen::Vector3f P;
        Eigen::Vector3f Q;
        for (int j = 0; j < 3; j++) {
            if (plane_normal[i][j] >= 0.0f) {
                P[j] = bmin[j];
                Q[j] = bmax[j];
            } else {
                P[j] = bmax[j];
                Q[j] = bmin[j];
            }
        }

        Eigen::Vector3f plane_point = { 0.0f, 0.0f, 0.0f };
        if (i == 4) {
            plane_point = { 0.0f, 0.0f, near };
            qtemp.vec() = plane_point;
            qtemp.w() = 0.0f;
            qtemp = orientation * qtemp * orientation.conjugate();
            plane_point = qtemp.vec();
        } else if (i == 5) {
            plane_point = { 0.0f, 0.0f, far };
            qtemp.vec() = plane_point;
            qtemp.w() = 0.0f;
            qtemp = orientation * qtemp * orientation.conjugate();
            plane_point = qtemp.vec();
        }
        Eigen::Vector3f tp = P - plane_point;
        Eigen::Vector3f tq = Q - plane_point;

        if (tp.dot(plane_normal[i]) <= 0.0f) {
            if (tq.dot(plane_normal[i]) > 0.0f) {
                return Containment::eIntersects;
            } else {
                return Containment::eDisjoint;
            }
        }
    }

    return Containment::eContains;
}

bool Ray::IntersectWithBbox(const BoundingBox &bbox, float &tmin) {
    Eigen::Vector3f pmin = bbox.center - bbox.extent;
    Eigen::Vector3f pmax = bbox.center + bbox.extent;
    float t0 = 0.0f;
    float t1 = std::numeric_limits<float>::max();
    for (int d = 0; d < 3; d++) {
        float tt0 = std::min((pmin[d] - origin[d]) / dir[d], (pmax[d] - origin[d]) / dir[d]);
        float tt1 = std::max((pmin[d] - origin[d]) / dir[d], (pmax[d] - origin[d]) / dir[d]);
        t0 = std::max(t0, tt0);
        t1 = std::min(t1, tt1);
        if (t0 > t1) return false;
    }
    tmin = t1;
    return true;
}

bool Ray::IntersectWithTriangle(const Eigen::Vector3f &v0, const Eigen::Vector3f &v1, const Eigen::Vector3f &v2,
    float &tmin) {
    Eigen::Vector3f e1 = v1 - v0;
    Eigen::Vector3f e2 = v2 - v0;
    Eigen::Vector3f s = origin - v0;

    float det = e1.cross(dir).dot(e2);
    if (det != 0) {
        float du = -s.cross(e2).dot(dir);
        float dv = e1.cross(dir).dot(s);
        float dt = -s.cross(e2).dot(e1);
        float u = du / det;
        float v = dv / det;
        float t = dt / det;
        if (u < 0 || v < 0 || 1 - u - v < 0) {
            return false;
        } else {
            tmin = t;
            return t >= 0;
        }
    } else {
        float t;
        if (DoesRayIntersectSegment(*this, v0, v1, t)) {
            tmin = t;
            return t >= 0;
        } else if (DoesRayIntersectSegment(*this, v0, v2, t)) {
            tmin = t;
            return t >= 0;
        } else if (DoesRayIntersectSegment(*this, v1, v2, t)) {
            tmin = t;
            return t >= 0;
        }
    }

    return false;
}

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

BoundingBox MathUtil::TransformBoundingBox(const Eigen::Matrix4f &mat, const BoundingBox &bbox) {
    Eigen::Vector3f p0 = TransformPoint(mat, { -bbox.extent.x(), -bbox.extent.y(), -bbox.extent.z() });
    Eigen::Vector3f p1 = TransformPoint(mat, { -bbox.extent.x(), -bbox.extent.y(),  bbox.extent.z() });
    Eigen::Vector3f p2 = TransformPoint(mat, { -bbox.extent.x(),  bbox.extent.y(), -bbox.extent.z() });
    Eigen::Vector3f p3 = TransformPoint(mat, { -bbox.extent.x(),  bbox.extent.y(),  bbox.extent.z() });
    Eigen::Vector3f p4 = TransformPoint(mat, {  bbox.extent.x(), -bbox.extent.y(), -bbox.extent.z() });
    Eigen::Vector3f p5 = TransformPoint(mat, {  bbox.extent.x(), -bbox.extent.y(),  bbox.extent.z() });
    Eigen::Vector3f p6 = TransformPoint(mat, {  bbox.extent.x(),  bbox.extent.y(), -bbox.extent.z() });
    Eigen::Vector3f p7 = TransformPoint(mat, {  bbox.extent.x(),  bbox.extent.y(),  bbox.extent.z() });
    Eigen::Vector3f max = p0.cwiseMax(p1).cwiseMax(p2).cwiseMax(p3).cwiseMax(p4).cwiseMax(p5).cwiseMax(p6).cwiseMax(p7);
    return { TransformPoint(mat, bbox.center), max };
}

Frustum MathUtil::TransformFrustum(const Eigen::Matrix4f &mat, const Frustum &frustum) {
    Eigen::Matrix3f rotation_mat;
    rotation_mat.col(0) = mat.col(0).head<3>().normalized();
    rotation_mat.col(1) = mat.col(1).head<3>().normalized();
    rotation_mat.col(2) = mat.col(2).head<3>().normalized();
    Eigen::Quaternionf rotation_q(rotation_mat);

    float sx = mat.col(0).head<3>().norm();
    float sy = mat.col(1).head<3>().norm();
    float sz = mat.col(2).head<3>().norm();
    float scale = std::max({ sx, sy, sz });

    Frustum res = frustum;
    res.orientation = rotation_q * frustum.orientation;
    res.origin = TransformPoint(mat, frustum.origin);
    res.near = scale * frustum.near;
    res.far = scale * frustum.far;

    return res;
}

Ray MathUtil::TransformRay(const Eigen::Matrix4f &mat, const Ray &ray) {
    return { TransformPoint(mat, ray.origin), TransformVector(mat, ray.dir) };
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