#include "CubemapRenderTarget.h"

CubemapRenderTarget::CubemapRenderTarget(const vku::Device *device, uint32_t width) : device(device), width(width) {
    BuildImages();
}

void CubemapRenderTarget::Begin(vk::CommandBuffer command_buffer, int face,
    const std::array<vk::ClearValue, 2> &clear_values) const {
    vk::Viewport viewport(0, 0, width, width, 0.0f, 1.0f);
    command_buffer.setViewport(0, { viewport });
    vk::Rect2D scissor({ 0, 0 }, { width, width });
    command_buffer.setScissor(0, { scissor });

    vk::RenderPassBeginInfo begin_info(render_pass, frame_buffers[face].get(), { { 0, 0 }, { width, width } },
        clear_values.size(), clear_values.data());
    command_buffer.beginRenderPass(begin_info, vk::SubpassContents::eInline);
}

void CubemapRenderTarget::End(vk::CommandBuffer command_buffer) const {
    command_buffer.endRenderPass();
}

void CubemapRenderTarget::InitialTransform(vk::CommandBuffer command_buffer) const {
    vk::ImageMemoryBarrier barrier(
        {}, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal,
        0, 0, color_image->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6 });
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {}, { barrier });
}
void CubemapRenderTarget::TransformToShaderRead(vk::CommandBuffer command_buffer) const {
    vk::ImageMemoryBarrier barrier(
        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        0, 0, color_image->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6 });
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {}, { barrier });
}

void CubemapRenderTarget::BuildAndWriteDescriptorSets(vk::DescriptorPool pool, vk::DescriptorSetLayout layout,
    vk::Sampler sampler) {
    vk::DescriptorSetAllocateInfo allocate_info(pool, 1, &layout);
    cubemap_descriptor_set = device->logical_device->allocateDescriptorSets(allocate_info)[0];

    vk::DescriptorImageInfo image_info(sampler, cubemap_image_view.get(), vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet write_info(cubemap_descriptor_set, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler,
        &image_info, nullptr, nullptr);
    device->logical_device->updateDescriptorSets({ write_info }, {});
}

void CubemapRenderTarget::BuildImages() {
    // image
    color_image = std::make_unique<vku::Image>(device, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm,
        vk::Extent3D(width, width, 1), 1, 6, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
        vk::ImageLayout::eUndefined, vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::ImageCreateFlagBits::eCubeCompatible);
    depth_image = std::make_unique<vku::Image>(device, vk::ImageType::e2D, vk::Format::eD24UnormS8Uint,
        vk::Extent3D(width, width, 1), 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageLayout::eUndefined,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    // image view
    vk::ImageViewCreateInfo cubemap_view_create_info({}, color_image->image.get(), vk::ImageViewType::eCube,
        vk::Format::eR8G8B8A8Unorm, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6 });
    cubemap_image_view = device->logical_device->createImageViewUnique(cubemap_view_create_info);
    color_image_views.resize(6);
    for (uint32_t i = 0; i < 6; i++) {
        vk::ImageViewCreateInfo color_view_create_info({}, color_image->image.get(), vk::ImageViewType::e2D,
            vk::Format::eR8G8B8A8Unorm, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, i, 1 });
        color_image_views[i] = device->logical_device->createImageViewUnique(color_view_create_info);
    }
    vk::ImageViewCreateInfo depth_view_create_info({}, depth_image->image.get(), vk::ImageViewType::e2D,
        vk::Format::eD24UnormS8Uint, {}, { vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 });
    depth_image_view = device->logical_device->createImageViewUnique(depth_view_create_info);
}

void CubemapRenderTarget::BuildFramebuffers(vk::RenderPass render_pass) {
    this->render_pass = render_pass;
    frame_buffers.resize(6);
    for (size_t i = 0; i < 6; i++) {
        std::array<vk::ImageView, 2> image_views = {
            color_image_views[i].get(),
            depth_image_view.get()
        };
        vk::FramebufferCreateInfo create_info({}, render_pass, image_views.size(), image_views.data(),
            width, width, 1);
        frame_buffers[i] = device->logical_device->createFramebufferUnique(create_info);
    }
}