#pragma once

#include "Eigen/Dense"
#include "VulkanBuffer.h"
#include "HostVisibleBuffer.h"

using namespace pepcy;

struct ObjectUB {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
};

struct PassUB {
    Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
};

struct Vertex {
    Eigen::Vector3f pos;
    Eigen::Vector3f color;

    static vk::VertexInputBindingDescription BindDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> AttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color))
        };
    }
};

struct FrameResources {
    FrameResources(const vku::Device *device, vk::DescriptorPool descriptor_pool,
        vk::DescriptorSetLayout obj_set_layout, vk::DescriptorSetLayout pass_set_layout, size_t n_object,
        size_t n_wave_vertices);

    std::vector<vk::DescriptorSet> obj_descriptor_set;
    vk::DescriptorSet pass_descriptor_set;
    std::unique_ptr<HostVisibleBuffer<ObjectUB>> obj_ub;
    std::unique_ptr<HostVisibleBuffer<PassUB>> pass_ub;
    std::unique_ptr<HostVisibleBuffer<Vertex>> wave_vb;
};


