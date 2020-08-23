#pragma once

#include "VulkanUtil.h"
#include "VulkanBuffer.h"
#include "HostVisibleBuffer.h"

using namespace pepcy;

struct alignas(64) ObjectUB {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f model_it = Eigen::Matrix4f::Identity();
};

struct alignas(64) VertPassUB {
    Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
};

struct alignas(64) FragPassUB {
    Eigen::Vector3f eye;
    float _0; // padding
    float near;
    float far;
    float delta_time;
    float total_time;
    Eigen::Vector4f ambient;
    Light lights[kMaxLights];
};

struct Vertex {
    Eigen::Vector3f pos;
    Eigen::Vector3f norm;

    static vk::VertexInputBindingDescription BindDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> AttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, norm))
        };
    }
};

struct FrameResources {
    FrameResources(const vku::Device *device, vk::DescriptorPool descriptor_pool,
        vk::DescriptorSetLayout obj_set_layout, size_t n_object,
        vk::DescriptorSetLayout mat_set_layout, size_t n_mat,
        vk::DescriptorSetLayout pass_set_layout, size_t n_pass,
        size_t n_wave_vertices);

    std::vector<vk::DescriptorSet> obj_set;
    std::vector<vk::DescriptorSet> mat_set;
    std::vector<vk::DescriptorSet> pass_set;
    std::unique_ptr<HostVisibleBuffer<ObjectUB>> obj_ub;
    std::unique_ptr<HostVisibleBuffer<MaterialUB>> mat_ub;
    std::unique_ptr<HostVisibleBuffer<VertPassUB>> vert_pass_ub;
    std::unique_ptr<HostVisibleBuffer<FragPassUB>> frag_pass_ub;
    std::unique_ptr<HostVisibleBuffer<Vertex>> wave_vb;
};


