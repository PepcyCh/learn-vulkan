#pragma once

#include <memory>

#include "VulkanImage.h"

using namespace pepcy;

class CubemapRenderTarget {
public:
    CubemapRenderTarget(const vku::Device *device, uint32_t width);

    void BuildFramebuffers(vk::RenderPass render_pass);
    void Begin(vk::CommandBuffer command_buffer, int face, const std::array<vk::ClearValue, 2> &clear_values) const;
    void End(vk::CommandBuffer command_buffer) const;

    void InitialTransform(vk::CommandBuffer command_buffer) const;
    void TransformToShaderRead(vk::CommandBuffer command_buffer) const;

    std::vector<vk::DescriptorPoolSize> DescriptorPoolSizes() const {
        return { vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1) };
    }
    constexpr uint32_t DescriptorSetCount() const {
        return 1;
    }
    void BuildAndWriteDescriptorSets(vk::DescriptorPool pool, vk::DescriptorSetLayout layout, vk::Sampler sampler);
    vk::DescriptorSet Cubemap() const {
        return cubemap_descriptor_set;
    }

private:
    void BuildImages();

    const vku::Device *device;
    vk::RenderPass render_pass;
    uint32_t width;

    std::vector<vk::UniqueFramebuffer> frame_buffers;
    std::unique_ptr<vku::Image> color_image;
    std::unique_ptr<vku::Image> depth_image;
    vk::UniqueImageView cubemap_image_view;
    std::vector<vk::UniqueImageView> color_image_views;
    vk::UniqueImageView depth_image_view;

    vk::DescriptorSet cubemap_descriptor_set;
};


