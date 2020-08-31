#include "BlurFilter.h"

#include <cmath>
#include <iostream>

BlurFilter::BlurFilter(const vku::Device *device, uint32_t width, uint32_t height) : device(device),
    width(width), height(height) {
    BuildLayouts();
    BuildImages();
}

void BlurFilter::OnResize(uint32_t new_width, uint32_t new_height) {
    if (width != new_width || height != new_height) {
        width = new_width;
        height = new_height;
        BuildImages();
        WriteDescriptorSets();
        first_transit = true;
    }
}

void BlurFilter::Execute(vk::Image input_image, int blur_time, vk::Pipeline hori_pipeline,
    vk::Pipeline vert_pipeline, vk::CommandBuffer command_buffer) {
    std::cerr << "BlurFilter::Execute" << std::endl;
    vk::ImageMemoryBarrier input_image_barrier(
        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead,
        vk::ImageLayout::ePresentSrcKHR, vk::ImageLayout::eTransferSrcOptimal,
        0, 0, input_image, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, { input_image_barrier });
    if (first_transit) {
        vk::ImageMemoryBarrier blur0_barrier(
            {}, vk::AccessFlagBits::eTransferWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
            0, 0, blur_image[0]->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, { blur0_barrier });
    } else {
        vk::ImageMemoryBarrier blur0_barrier(
            vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eTransferWrite,
            vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eTransferDstOptimal,
            0, 0, blur_image[0]->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, { blur0_barrier });
    }
    vk::ImageCopy copy_region({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 }, { 0, 0, 0 },
        { vk::ImageAspectFlagBits::eColor, 0, 0, 1 }, { 0, 0, 0 }, { width, height, 1 });
    command_buffer.copyImage(input_image, vk::ImageLayout::eTransferSrcOptimal,
        blur_image[0]->image.get(), vk::ImageLayout::eTransferDstOptimal, { copy_region });

    auto weights = GaussWeights(2.5f);
    int blur_rad = weights.size() / 2;

    command_buffer.pushConstants(compute_pipeline_layout.get(), vk::ShaderStageFlagBits::eCompute,
        0, 1 * 4, &blur_rad);
    command_buffer.pushConstants(compute_pipeline_layout.get(), vk::ShaderStageFlagBits::eCompute,
        4, weights.size() * 4, weights.data());

    vk::ImageMemoryBarrier pre_barrier(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral,
        0, 0, blur_image[0]->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {}, { pre_barrier });

    if (first_transit) {
        pre_barrier = vk::ImageMemoryBarrier(
            {}, vk::AccessFlagBits::eShaderWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            0, 0, blur_image[1]->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {}, { pre_barrier });
        first_transit = false;
    }

    for (int i = 0; i < blur_time; i++) {
        uint32_t n_group_x = (width + 255) >> 8;
        uint32_t n_group_y = height;
        command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, hori_pipeline);
        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute_pipeline_layout.get(), 0,
            { compute_descriptor_set[0] }, {});
        command_buffer.dispatch(n_group_x, n_group_y, 1);

        std::array<vk::ImageMemoryBarrier, 2> barriers1 = {
            vk::ImageMemoryBarrier(
                vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eShaderWrite,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                0, 0, blur_image[0]->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }),
            vk::ImageMemoryBarrier(
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                0, 0, blur_image[1]->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }),
        };
        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {}, barriers1);


        n_group_x = width;
        n_group_y = (height + 255) >> 8;
        command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, vert_pipeline);
        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute_pipeline_layout.get(), 0,
            { compute_descriptor_set[1] }, {});
        command_buffer.dispatch(n_group_x, n_group_y, 1);

        std::array<vk::ImageMemoryBarrier, 2> barriers2 = {
            vk::ImageMemoryBarrier(
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                0, 0, blur_image[0]->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }),
            vk::ImageMemoryBarrier(
                vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eShaderWrite,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                0, 0, blur_image[1]->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }),
        };
        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {}, barriers2);
    }

    vk::ImageMemoryBarrier after_barriers(
        vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
        0, 0, blur_image[0]->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, { after_barriers });
    input_image_barrier = vk::ImageMemoryBarrier(
        vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eTransferWrite,
        vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eTransferDstOptimal,
        0, 0, input_image, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, { input_image_barrier });
    command_buffer.copyImage(blur_image[0]->image.get(), vk::ImageLayout::eTransferSrcOptimal,
        input_image, vk::ImageLayout::eTransferDstOptimal, { copy_region });

    input_image_barrier = vk::ImageMemoryBarrier(
        vk::AccessFlagBits::eTransferWrite, {},
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
        0, 0, input_image, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe,
        {}, {}, {}, { input_image_barrier });
}

void BlurFilter::BuildDescriptorSets(vk::DescriptorPool pool) {
    std::vector<vk::DescriptorSetLayout> layouts(2, compute_descriptor_set_layouts[0].get());
    vk::DescriptorSetAllocateInfo allocate_info(pool, layouts.size(), layouts.data());
    compute_descriptor_set = device->logical_device->allocateDescriptorSets(allocate_info);
}

void BlurFilter::WriteDescriptorSets() {
    std::array<vk::DescriptorImageInfo, 2> image_infos = {
        vk::DescriptorImageInfo({}, blur_image_view[0].get(), vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, blur_image_view[1].get(), vk::ImageLayout::eGeneral),
    };

    std::array<vk::WriteDescriptorSet, 4> writes = {
        vk::WriteDescriptorSet(compute_descriptor_set[0], 0, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[0], nullptr, nullptr),
        vk::WriteDescriptorSet(compute_descriptor_set[0], 1, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[1], nullptr, nullptr),
        vk::WriteDescriptorSet(compute_descriptor_set[1], 0, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[1], nullptr, nullptr),
        vk::WriteDescriptorSet(compute_descriptor_set[1], 1, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[0], nullptr, nullptr),
    };

    device->logical_device->updateDescriptorSets(writes, {});
}

void BlurFilter::BuildLayouts() {
    compute_descriptor_set_layouts.resize(1);

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute),
    };
    vk::DescriptorSetLayoutCreateInfo set_layout_create_info({}, bindings.size(), bindings.data());
    compute_descriptor_set_layouts[0] = device->logical_device->createDescriptorSetLayoutUnique(
        set_layout_create_info);

    std::array<vk::PushConstantRange, 1> push_constants = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, 12 * 4)
    };

    auto layouts = vk::uniqueToRaw(compute_descriptor_set_layouts);
    vk::PipelineLayoutCreateInfo pipeline_layout_create_info({}, layouts.size(), layouts.data(),
        push_constants.size(), push_constants.data());
    compute_pipeline_layout = device->logical_device->createPipelineLayoutUnique(pipeline_layout_create_info);
}

void BlurFilter::BuildImages() {
    blur_image[0] = std::make_unique<vku::Image>(device, vk::ImageType::e2D, vk::Format::eB8G8R8A8Unorm,
        vk::Extent3D(width, height, 1), 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eStorage, vk::ImageLayout::eUndefined,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    blur_image[1] = std::make_unique<vku::Image>(device, vk::ImageType::e2D, vk::Format::eB8G8R8A8Unorm,
        vk::Extent3D(width, height, 1), 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage, vk::ImageLayout::eUndefined,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::ImageViewCreateInfo image_view_create_info({}, blur_image[0]->image.get(), vk::ImageViewType::e2D,
        vk::Format::eR8G8B8A8Unorm, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    blur_image_view[0] = device->logical_device->createImageViewUnique(image_view_create_info);

    image_view_create_info.setImage(blur_image[1]->image.get());
    blur_image_view[1] = device->logical_device->createImageViewUnique(image_view_create_info);
}

std::vector<float> BlurFilter::GaussWeights(float sigma) {
    float two_sigma_sqr = 2.0f * sigma * sigma;
    int blur_rad = std::min((int) std::ceil(2.0f * sigma), kMaxBlurRadius);
    std::vector<float> weights(2 * blur_rad + 1);

    float sum = 0;
    for (int i = -blur_rad; i <= blur_rad; i++) {
        float x = i * i;
        weights[i + blur_rad] = std::exp(-x / two_sigma_sqr);
        sum += weights[i + blur_rad];
    }
    for (float &w : weights) {
        w /= sum;
    }

    return weights;
}