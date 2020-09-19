#pragma once

#include "MathUtil.h"

class Camera {
public:
    Camera();

    void SetLens(float fov, float aspect, float near, float far, bool flip_y = true);
    void SetPosition(const Eigen::Vector3f &pos);
    void LookAt(const Eigen::Vector3f &pos, const Eigen::Vector3f &lookat, const Eigen::Vector3f &up);

    void Strafe(float d);
    void Walk(float d);

    void Pitch(float angle);
    void RotateY(float angle);

    void UpdateViewMatrix();

    Eigen::Matrix4f View() const {
        return view;
    }
    Eigen::Matrix4f Proj() const {
        return proj;
    }

    Eigen::Vector3f Position() const {
        return position;
    }
    Eigen::Vector3f Right() const {
        return right;
    }
    Eigen::Vector3f Up() const {
        return up;
    }
    Eigen::Vector3f Look() const {
        return look;
    }

    float Near() const {
        return near;
    }
    float Far() const {
        return far;
    }
    float Aspect() const {
        return aspect;
    }
    float Fov() const {
        return fov;
    }

    float NearHeight() const {
        return near_height;
    }
    float NearWidth() const {
        return near_height * aspect;
    }
    float FarHeight() const {
        return far_height;
    }
    float FarWidth() const {
        return far_height * aspect;
    }

private:
    Eigen::Vector3f position = { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f right = { 1.0f, 0.0f, 0.0f };
    Eigen::Vector3f up = { 0.0f, 1.0f, 0.0f };
    Eigen::Vector3f look = { 0.0f, 0.0f, 1.0f };

    float near = 0.0f;
    float far = 0.0f;
    float aspect = 0.0f;
    float fov = 0.0f;
    float near_height = 0.0f;
    float far_height = 0.0f;

    bool view_dirty = true;

    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
};


