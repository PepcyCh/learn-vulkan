#include "Camera.h"

Camera::Camera() {
    SetLens(MathUtil::kPiDiv4, 1.0f, 0.1f, 500.0f);
}

void Camera::SetLens(float fov, float aspect, float near, float far) {
    this->fov = fov;
    this->aspect = aspect;
    this->near = near;
    this->far = far;
    proj = MathUtil::Perspective(fov, aspect, near, far, true);
}

void Camera::SetPosition(const Eigen::Vector3f &pos) {
    position = pos;
    view_dirty = true;
}

void Camera::LookAt(const Eigen::Vector3f &pos, const Eigen::Vector3f &lookat, const Eigen::Vector3f &up) {
    position = pos;
    look = (lookat - pos).normalized();
    right = look.cross(up).normalized();
    this->up = right.cross(look);
    view_dirty = true;
}

void Camera::Strafe(float d) {
    position += right * d;
    view_dirty = true;
}
void Camera::Walk(float d) {
    position += look * d;
    view_dirty = true;
}

void Camera::Pitch(float angle) {
    Eigen::Matrix4f mat = MathUtil::AngelAxis(angle, right);
    look = MathUtil::TransformVector(mat, look);
    up = MathUtil::TransformVector(mat, up);
    view_dirty = true;
}
void Camera::RotateY(float angle) {
    Eigen::Matrix4f mat = MathUtil::AngelAxis(angle, { 0.0f, 1.0f, 0.0f });
    look = MathUtil::TransformVector(mat, look);
    up = MathUtil::TransformVector(mat, up);
    right = MathUtil::TransformVector(mat, right);
    view_dirty = true;
}

void Camera::UpdateViewMatrix() {
    if (view_dirty) {
        right.normalize();
        up.normalize();
        look.normalize();
        view = MathUtil::ViewTransform(position, right, up, -look);
        view_dirty = false;
    }
}