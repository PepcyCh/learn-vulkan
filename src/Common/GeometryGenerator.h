#pragma once

#include <vector>

#include "Eigen/Dense"

class GeometryGenerator {
public:
    struct Vertex {
        Vertex() = default;
        Vertex(const Eigen::Vector3f &p, const Eigen::Vector3f &n, const Eigen::Vector3f &t,
            const Eigen::Vector2f &uv) : pos(p), norm(n), tan(t), texc(uv) {}
        Vertex(float px, float py, float pz, float nx, float ny, float nz, float tx, float ty, float tz,
            float u, float v) : pos(px, py, pz), norm(nx, ny, nz), tan(tx, ty, tz), texc(u, v) {}

        Eigen::Vector3f pos;
        Eigen::Vector3f norm;
        Eigen::Vector3f tan;
        Eigen::Vector2f texc;
    };

    struct MeshData {
        std::vector<uint16_t> &GetIndices16() {
            if (indices16.empty()) {
                indices16.resize(indices32.size());
                for (int i = 0; i < indices32.size(); i++) {
                    indices16[i] = indices32[i];
                }
            }
            return indices16;
        }
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices32;

    private:
        std::vector<uint16_t> indices16;
    };

    MeshData Box(float w, float h, float d, uint32_t n_subdiv);
    MeshData Sphere(float radius, uint32_t n_slice, uint32_t n_stack);
    MeshData Geosphere(float radius, uint32_t n_subdiv);
    MeshData Cylinder(float bottom_radius, float top_radius, float h, uint32_t n_slice, uint32_t n_stack);
    MeshData Grid(float w, float d, uint32_t n, uint32_t m);
    MeshData Quad(float x, float y, float w, float h, float d);

private:
    void Subdivide(MeshData &mesh);
    Vertex Midpoint(const Vertex &v0, const Vertex &v1);
    void BuildCylinderBottomCap(float br, float tr, float h, uint32_t n_slice, uint32_t n_stack, MeshData &mesh);
    void BuildCylinderTopCap(float br, float tr, float h, uint32_t n_slice, uint32_t n_stack, MeshData &mesh);
};