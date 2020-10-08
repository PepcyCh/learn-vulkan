#include "DepthRenderTarget.h"

DepthRenderTarget::DepthRenderTarget(const vku::Device *device, uint32_t width, uint32_t height) : device(device),
    width(width), height(height) {
    BuildImages();
}

void DepthRenderTarget::Begin(vk::CommandBuffer command_buffer,
    const std::array<vk::ClearValue, 1> &clear_values) const {
    vk::RenderPassBeginInfo begin_info(render_pass, frame_buffer.get(), { { 0, 0 }, { width, height } },
        clear_values.size(), clear_values.data());
    command_buffer.beginRenderPass(begin_info, vk::SubpassContents::eInline);

    vk::Viewport viewport(0, 0, width, height, 0.0f, 1.0f);
    command_buffer.setViewport(0, { viewport });
    vk::Rect2D scissor({ 0, 0 }, { width, height });
    command_buffer.setScissor(0, { scissor });
}

void DepthRenderTarget::End(vk::CommandBuffer command_buffer) const {
    command_buffer.endRenderPass();
}

void DepthRenderTarget::BuildAndWriteDescriptorSets(vk::DescriptorPool pool, vk::DescriptorSetLayout layout,
    vk::Sampler sampler) {
    vk::DescriptorSetAllocateInfo allocate_info(pool, 1, &layout);
    descriptor_set = device->logical_device->allocateDescriptorSets(allocate_info)[0];

    vk::DescriptorImageInfo image_info(sampler, depth_image_view.get(), vk::ImageLayout::eDepthStencilReadOnlyOptimal);
    vk::WriteDescriptorSet write_info(descriptor_set, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler,
        &image_info, nullptr, nullptr);
    device->logical_device->updateDescriptorSets({ write_info }, {});
}

void DepthRenderTarget::BuildImages() {
    depth_image = std::make_unique<vku::Image>(device, vk::ImageType::e2D, vk::Format::eD24UnormS8Uint,
        vk::Extent3D(width, height, 1), 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout::eUndefined, vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::ImageViewCreateInfo depth_view_create_info({}, depth_image->image.get(), vk::ImageViewType::e2D,
        vk::Format::eD24UnormS8Uint, {}, { vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 });
    depth_image_view = device->logical_device->createImageViewUnique(depth_view_create_info);
}

void DepthRenderTarget::BuildFramebuffers(vk::RenderPass render_pass) {
    this->render_pass = render_pass;
    vk::FramebufferCreateInfo create_info({}, render_pass, 1, &depth_image_view.get(), width, height, 1);
    frame_buffer = device->logical_device->createFramebufferUnique(create_info);
}