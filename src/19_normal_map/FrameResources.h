#pragma once

#include "VulkanUtil.h"
#include "VulkanBuffer.h"
#include "HostVisibleBuffer.h"

using namespace pepcy;

struct alignas(64) ObjectUB {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f model_it = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f tex_transform = Eigen::Matrix4f::Identity();
    int mat_index;
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
    Eigen::Vector4f ambient = { 0.0f, 0.0f, 0.0f, 1.0f };
    Eigen::Vector4f fog_color = { 0.7f, 0.7f, 0.7f, 1.0f };
    float fog_start = 25.0f;
    float fog_end = 150.0f;
    Eigen::Vector2f _1; // padding
    Light lights[kMaxLights];
};

struct MaterialData {
    Eigen::Vector4f albedo = { 0.0f, 0.0f, 0.0f, 1.0f };
    Eigen::Vector3f fresnel_r0 = { 0.0f, 0.0f, 0.0f };
    float roughness = 0.0f;
    Eigen::Matrix4f mat_transform = Eigen::Matrix4f::Identity();
    uint32_t diffuse_index = 0;
    uint32_t normal_index = 0;
};

struct Vertex {
    Eigen::Vector3f pos;
    Eigen::Vector3f norm;
    Eigen::Vector2f texc;
    Eigen::Vector3f tan;

    static vk::VertexInputBindingDescription BindDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 4> AttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, norm)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texc)),
            vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, tan)),
        };
    }
};

struct FrameResources {
    FrameResources(const vku::Device *device, vk::DescriptorPool descriptor_pool,
        vk::DescriptorSetLayout obj_set_layout, size_t n_object,
        vk::DescriptorSetLayout tex_set_layout, vk::DescriptorSetLayout cube_set_layout,
        vk::DescriptorSetLayout mat_set_layout, size_t n_mat,
        vk::DescriptorSetLayout pass_set_layout, size_t n_pass);

    std::vector<vk::DescriptorSet> obj_set;
    std::vector<vk::DescriptorSet> tex_set;
    std::vector<vk::DescriptorSet> mat_set;
    std::vector<vk::DescriptorSet> pass_set;

    std::unique_ptr<HostVisibleBuffer<ObjectUB>> obj_ub;
    std::unique_ptr<HostVisibleBuffer<MaterialData>> mat_ub;
    std::unique_ptr<HostVisibleBuffer<VertPassUB>> vert_pass_ub;
    std::unique_ptr<HostVisibleBuffer<FragPassUB>> frag_pass_ub;
};