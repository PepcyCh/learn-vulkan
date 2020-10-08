#pragma once

#include <memory>

#include "VulkanImage.h"

using namespace pepcy;

class DepthRenderTarget {
public:
    DepthRenderTarget(const vku::Device *device, uint32_t width, uint32_t height);

    void BuildFramebuffers(vk::RenderPass render_pass);
    void Begin(vk::CommandBuffer command_buffer, const std::array<vk::ClearValue, 1> &clear_values) const;
    void End(vk::CommandBuffer command_buffer) const;

    std::vector<vk::DescriptorPoolSize> DescriptorPoolSizes() const {
        return { vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1) };
    }
    constexpr uint32_t DescriptorSetCount() const {
        return 1;
    }
    void BuildAndWriteDescriptorSets(vk::DescriptorPool pool, vk::DescriptorSetLayout layout, vk::Sampler sampler);
    vk::DescriptorSet ShadowMap() const {
        return descriptor_set;
    }

private:
    void BuildImages();

    const vku::Device *device;
    vk::RenderPass render_pass;
    uint32_t width;
    uint32_t height;

    vk::UniqueFramebuffer frame_buffer;
    std::unique_ptr<vku::Image> depth_image;
    vk::UniqueImageView depth_image_view;

    vk::DescriptorSet descriptor_set;
};


