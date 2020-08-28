#pragma once

#include <vector>

#include "Eigen/Dense"

class Wave {
public:
    Wave(int m, int n, float dx, float dt, float speed, float damping);
    Wave(const Wave &rhs) = delete;
    Wave &operator=(const Wave &rhs) = delete;

    int RowCount() const {
        return n_row;
    }
    int ColumnCount() const {
        return n_col;
    }
    int VertexCount() const {
        return n_vertex;
    }
    int TriangleCount() const {
        return n_triangle;
    }
    float Width() const {
        return n_col * spatial_step;
    }
    float Depth() const {
        return n_row * spatial_step;
    }

    const Eigen::Vector3f &Position(int i) const {
        return curr_solution[i];
    }
    const Eigen::Vector3f &Normal(int i) const {
        return normals[i];
    }
    const Eigen::Vector3f &Tanget(int i) const {
        return tangents[i];
    }

    void Update(float dt);
    void Disturb(int i, int j, float magnitude);

private:
    int n_row = 0;
    int n_col = 0;
    int n_vertex = 0;
    int n_triangle = 0;

    float k1 = 0.0f;
    float k2 = 0.0f;
    float k3 = 0.0f;

    float time_step = 0.0f;
    float spatial_step = 0.0f;

    std::vector<Eigen::Vector3f> prev_solution;
    std::vector<Eigen::Vector3f> curr_solution;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> tangents;
};