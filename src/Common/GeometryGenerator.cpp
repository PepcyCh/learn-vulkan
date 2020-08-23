#include "GeometryGenerator.h"

#include <cmath>
#include <algorithm>

#include "MathUtil.h"

GeometryGenerator::MeshData
GeometryGenerator::Box(float w, float h, float d, uint32_t n_subdiv) {
    MeshData mesh;

    float hw = w * 0.5f;
    float hh = h * 0.5f;
    float hd = d * 0.5f;

    Vertex v[24];

    v[0] = Vertex(-hw, -hh, -hd, 0.0f, 0.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f);
    v[1] = Vertex(-hw,  hh, -hd, 0.0f, 0.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    v[2] = Vertex( hw,  hh, -hd, 0.0f, 0.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    v[3] = Vertex( hw, -hh, -hd, 0.0f, 0.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    v[4] = Vertex(-hw, -hh,  hd, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    v[5] = Vertex( hw, -hh,  hd, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f);
    v[6] = Vertex( hw,  hh,  hd, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    v[7] = Vertex(-hw,  hh,  hd, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    v[8]  = Vertex(-hw, -hh, -hd, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    v[9]  = Vertex( hw, -hh, -hd, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f);
    v[10] = Vertex( hw, -hh,  hd, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    v[11] = Vertex(-hw, -hh,  hd, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f);

    v[12] = Vertex(-hw,  hh, -hd, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f);
    v[13] = Vertex(-hw,  hh,  hd, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f);
    v[14] = Vertex( hw,  hh,  hd, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f);
    v[15] = Vertex( hw,  hh, -hd, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f);

    v[16] = Vertex(-hw, -hh, -hd, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f);
    v[17] = Vertex(-hw, -hh,  hd, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    v[18] = Vertex(-hw,  hh,  hd, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f);
    v[19] = Vertex(-hw,  hh, -hd, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    v[20] = Vertex( hw, -hh, -hd, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f);
    v[21] = Vertex( hw,  hh, -hd, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f);
    v[22] = Vertex( hw,  hh,  hd, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f);
    v[23] = Vertex( hw, -hh,  hd, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f);

    mesh.vertices.assign(v, v + 24);

    uint32_t ind[36] = {
        0, 1, 2,    0, 2, 3,    4, 5, 6,    4, 6, 7,
        8, 9, 10,   8, 10, 11,  12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23
    };
    mesh.indices32.assign(ind, ind + 36);

    n_subdiv = std::min(n_subdiv, 6u);
    for (uint32_t i = 0; i < n_subdiv; i++) {
        Subdivide(mesh);
    }

    return mesh;
}

GeometryGenerator::MeshData
GeometryGenerator::Sphere(float radius, uint32_t n_slice, uint32_t n_stack) {
    MeshData mesh;

    Vertex top(0.0f, radius, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    Vertex bottom(0.0f, -radius, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    mesh.vertices.push_back(top);

    float dphi = MathUtil::kPi / n_stack;
    float dtheta = 2.0f * MathUtil::kPi / n_slice;

    for (uint32_t i = 1; i < n_stack; i++) {
        float phi = dphi * i;
        for (uint32_t j = 0; j <= n_slice; j++) {
            float theta = dtheta * j;
            float stheta = std::sin(theta);
            float ctheta = std::cos(theta);
            float sphi = std::sin(phi);
            float cphi = std::cos(phi);

            Vertex v;
            v.pos.x() = radius * sphi * ctheta;
            v.pos.y() = radius * cphi;
            v.pos.z() = radius * sphi * stheta;
            v.tan.x() = -stheta;
            v.tan.y() = 0.0f;
            v.tan.z() = ctheta;
            v.norm = v.pos.normalized();
            v.texc.x() = theta / MathUtil::k2Pi;
            v.texc.y() = phi / MathUtil::kPi;

            mesh.vertices.push_back(v);
        }
    }

    mesh.vertices.push_back(bottom);

    for (uint32_t i = 1; i <= n_slice; i++) {
        mesh.indices32.push_back(0);
        mesh.indices32.push_back(i + 1);
        mesh.indices32.push_back(i);
    }
    uint32_t base = 1;
    uint32_t n_ring = n_slice + 1;
    for (uint32_t i = 0; i < n_stack - 2; i++) {
        for (uint32_t j = 0; j < n_slice; j++) {
            mesh.indices32.push_back(base + i * n_ring + j);
            mesh.indices32.push_back(base + i * n_ring + j + 1);
            mesh.indices32.push_back(base + (i + 1) * n_ring + j + 1);
            mesh.indices32.push_back(base + i * n_ring + j);
            mesh.indices32.push_back(base + (i + 1) * n_ring + j + 1);
            mesh.indices32.push_back(base + (i + 1) * n_ring + j);
        }
    }
    uint32_t bottom_i = (uint32_t) mesh.vertices.size() - 1;
    base = bottom_i - n_ring;
    for (uint32_t i = 0; i < n_slice; i++) {
        mesh.indices32.push_back(bottom_i);
        mesh.indices32.push_back(base + i);
        mesh.indices32.push_back(base + i + 1);
    }

    return mesh;
}

GeometryGenerator::MeshData
GeometryGenerator::Geosphere(float radius, uint32_t n_subdiv) {
    MeshData mesh;

    // X^2 + Z^2 = 1
    // Z / X = (sqrt(5) + 1) / 2
    const float X = 0.525731f;
    const float Z = 0.850651f;
    Eigen::Vector3f pos[12] = {
        Eigen::Vector3f(-X, 0.0f, Z),  Eigen::Vector3f(X, 0.0f, Z),
        Eigen::Vector3f(-X, 0.0f, -Z), Eigen::Vector3f(X, 0.0f, -Z),
        Eigen::Vector3f(0.0f, Z, X),   Eigen::Vector3f(0.0f, Z, -X),
        Eigen::Vector3f(0.0f, -Z, X),  Eigen::Vector3f(0.0f, -Z, -X),
        Eigen::Vector3f(Z, X, 0.0f),   Eigen::Vector3f(-Z, X, 0.0f),
        Eigen::Vector3f(Z, -X, 0.0f),  Eigen::Vector3f(-Z, -X, 0.0f)
    };
    uint32_t ind[60] = {
        1, 4, 0,  4, 9, 0,  4, 5, 9,  8, 5, 4,  1, 8, 4,
        1, 10, 8, 10, 3, 8, 8, 3, 5,  3, 2, 5,  3, 7, 2,
        3, 10, 7, 10, 6, 7, 6, 11, 7, 6, 0, 11, 6, 1, 0,
        10, 1, 6, 11, 0, 9, 2, 11, 9, 5, 2, 9,  11, 2, 7
    };
    mesh.indices32.assign(ind, ind + 60);

    mesh.vertices.resize(12);
    for (int i = 0; i < 12; i++) {
        mesh.vertices[i].pos = pos[i];
    }

    n_subdiv = std::min(n_subdiv, 6u);
    for (uint32_t i = 0; i < n_subdiv; i++) {
        Subdivide(mesh);
    }

    for (auto &vertex : mesh.vertices) {
        vertex.norm = vertex.pos.normalized();
        vertex.pos = vertex.norm * radius;

        float theta = std::atan2(vertex.pos.z(), vertex.pos.x());
        if (theta < 0) {
            theta += 2.0f * MathUtil::k2Pi;
        }
        float phi = std::acos(vertex.pos.y() / radius);
        vertex.texc.x() = theta / MathUtil::k2Pi;
        vertex.texc.y() = phi / MathUtil::kPi;
        vertex.tan.x() = -std::sin(theta);
        vertex.tan.y() = 0;
        vertex.tan.z() = std::cos(theta);
    }

    return mesh;
}

GeometryGenerator::MeshData
GeometryGenerator::Cylinder(float bottom_radius, float top_radius, float h, uint32_t n_slice, uint32_t n_stack) {
    MeshData mesh;

    float dh = h / n_stack;
    float dr = (top_radius - bottom_radius) / n_stack;

    for (uint32_t i = 0; i <= n_stack; i++) {
        float y = -0.5f * h + dh * i;
        float r = bottom_radius + dr * i;
        float dtheta = MathUtil::k2Pi / n_slice;
        for (uint32_t j = 0; j <= n_slice; j++) {
            float theta = dtheta * j;
            float s = std::sin(theta);
            float c = std::cos(theta);

            Vertex v;
            v.pos = Eigen::Vector3f(r * c, y, r * s);
            v.tan = Eigen::Vector3f(-s, 0.0f, c);
            v.texc.x() = (float) j / n_slice;
            v.texc.y() = 1.0f - (float) i / n_stack;

            float tr = bottom_radius - top_radius;
            Eigen::Vector3f bitan(tr * c, -h, tr * s);
            v.norm = (v.tan.cross(bitan)).normalized();

            mesh.vertices.push_back(v);
        }
    }

    uint32_t n_ring = n_slice + 1;
    for (int i = 0; i < n_stack; i++) {
        for (int j = 0; j < n_slice; j++) {
            mesh.indices32.push_back(i * n_ring + j);
            mesh.indices32.push_back((i + 1) * n_ring + j);
            mesh.indices32.push_back((i + 1) * n_ring + j + 1);
            mesh.indices32.push_back(i * n_ring + j);
            mesh.indices32.push_back((i + 1) * n_ring + j + 1);
            mesh.indices32.push_back(i * n_ring + j + 1);
        }
    }

    BuildCylinderBottomCap(bottom_radius, top_radius, h, n_slice, n_stack, mesh);
    BuildCylinderTopCap(bottom_radius, top_radius, h, n_slice, n_stack, mesh);

    return mesh;
}

GeometryGenerator::MeshData
GeometryGenerator::Grid(float w, float d, uint32_t n, uint32_t m) {
    MeshData mesh;

    uint32_t n_vertex = n * m;
    uint32_t n_face = 2 * (n - 1) * (m - 1);

    float hw = 0.5f * w;
    float hd = 0.5f * d;
    float du = 1.0f / (n - 1);
    float dv = 1.0f / (m - 1);
    float dx = w * du;
    float dz = d * dv;

    mesh.vertices.resize(n_vertex);
    for (uint32_t i = 0; i < m; i++) {
        float z = hd - dz * i;
        for (uint32_t j = 0; j < n; j++) {
            float x = -hw + dx * j;

            uint32_t id = i * n + j;
            mesh.vertices[id].pos = Eigen::Vector3f(x, 0.0f, z);
            mesh.vertices[id].norm = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
            mesh.vertices[id].tan = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
            mesh.vertices[id].texc = Eigen::Vector2f(j * du, i * dv);
        }
    }

    mesh.indices32.resize(n_face * 3);
    uint32_t k = 0;
    for (uint32_t i = 0; i < m - 1; i++) {
        for (uint32_t j = 0; j < n - 1; j++) {
            mesh.indices32[k] = i * n + j;
            mesh.indices32[k + 1] = i * n + j + 1;
            mesh.indices32[k + 2] = (i + 1) * n + j;
            mesh.indices32[k + 3] = i * n + j + 1;
            mesh.indices32[k + 4] = (i + 1) * n + j + 1;
            mesh.indices32[k + 5] = (i + 1) * n + j;
            k += 6;
        }
    }

    return mesh;
}

GeometryGenerator::MeshData
GeometryGenerator::Quad(float x, float y, float w, float h, float d) {
    MeshData mesh;

    mesh.vertices.resize(4);
    mesh.indices32.resize(6);

    mesh.vertices[0] = Vertex(x, y, d, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    mesh.vertices[1] = Vertex(x, y - h, d, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    mesh.vertices[2] = Vertex(x + w, y - h, d, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f);
    mesh.vertices[3] = Vertex(x + w, y, d, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f);

    mesh.indices32[0] = 0;
    mesh.indices32[1] = 1;
    mesh.indices32[2] = 2;
    mesh.indices32[3] = 0;
    mesh.indices32[4] = 2;
    mesh.indices32[5] = 3;

    return mesh;
}

void GeometryGenerator::Subdivide(MeshData &mesh) {
    MeshData copy = mesh;
    mesh.vertices.clear();
    mesh.indices32.clear();

    uint32_t n_tri = copy.indices32.size() / 3;
    for (uint32_t i = 0; i < n_tri; i++) {
        Vertex v0 = copy.vertices[copy.indices32[i * 3]];
        Vertex v1 = copy.vertices[copy.indices32[i * 3 + 1]];
        Vertex v2 = copy.vertices[copy.indices32[i * 3 + 2]];
        Vertex m0 = Midpoint(v1, v2);
        Vertex m1 = Midpoint(v2, v0);
        Vertex m2 = Midpoint(v0, v1);

        mesh.vertices.push_back(v0);
        mesh.vertices.push_back(v1);
        mesh.vertices.push_back(v2);
        mesh.vertices.push_back(m0);
        mesh.vertices.push_back(m1);
        mesh.vertices.push_back(m2);

        mesh.indices32.push_back(i * 6);
        mesh.indices32.push_back(i * 6 + 5);
        mesh.indices32.push_back(i * 6 + 4);
        mesh.indices32.push_back(i * 6 + 5);
        mesh.indices32.push_back(i * 6 + 1);
        mesh.indices32.push_back(i * 6 + 3);
        mesh.indices32.push_back(i * 6 + 4);
        mesh.indices32.push_back(i * 6 + 3);
        mesh.indices32.push_back(i * 6 + 2);
        mesh.indices32.push_back(i * 6 + 5);
        mesh.indices32.push_back(i * 6 + 3);
        mesh.indices32.push_back(i * 6 + 4);
    }
}

GeometryGenerator::Vertex GeometryGenerator::Midpoint(const Vertex &v0, const Vertex &v1) {
    Vertex v;
    v.pos = 0.5f * (v0.pos + v1.pos);
    v.norm = (v0.norm + v1.norm).normalized();
    v.tan = (v0.tan + v1.tan).normalized();
    v.texc = 0.5f * (v0.texc + v1.texc);
    return v;
}

void GeometryGenerator::BuildCylinderBottomCap(float br, float tr, float h, uint32_t n_slice, uint32_t n_stack,
    MeshData &mesh) {
    uint32_t base = mesh.vertices.size();
    float dtheta = MathUtil::k2Pi / n_slice;
    float y = -h * 0.5f;
    for (uint32_t i = 0; i <= n_slice; i++) {
        float theta = dtheta * i;
        float x = br * std::cos(theta);
        float z = br * std::sin(theta);
        float u = x / h + 0.5f;
        float v = z / h + 0.5f;
        mesh.vertices.emplace_back(x, y, z, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, u, v);
    }
    mesh.vertices.emplace_back(0.0f, y, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f);
    uint32_t center = mesh.vertices.size() - 1;
    for (uint32_t i = 0; i < n_slice; i++) {
        mesh.indices32.push_back(center);
        mesh.indices32.push_back(base + i);
        mesh.indices32.push_back(base + i + 1);
    }
}

void GeometryGenerator::BuildCylinderTopCap(float br, float tr, float h, uint32_t n_slice, uint32_t n_stack,
    MeshData &mesh) {
    uint32_t base = mesh.vertices.size();
    float dtheta = MathUtil::k2Pi / n_slice;
    float y = h * 0.5f;
    for (uint32_t i = 0; i <= n_slice; i++) {
        float theta = dtheta * i;
        float x = tr * std::cos(theta);
        float z = tr * std::sin(theta);
        float u = x / h + 0.5f;
        float v = z / h + 0.5f;
        mesh.vertices.emplace_back(x, y, z, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, u, v);
    }
    mesh.vertices.emplace_back(0.0f, y, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f);
    uint32_t center = mesh.vertices.size() - 1;
    for (uint32_t i = 0; i < n_slice; i++) {
        mesh.indices32.push_back(center);
        mesh.indices32.push_back(base + i + 1);
        mesh.indices32.push_back(base + i);
    }
}