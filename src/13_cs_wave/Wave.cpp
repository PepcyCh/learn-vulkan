#include "Wave.h"

#include <algorithm>

Wave::Wave(const vku::Device *device, int m, int n, float dx, float dt, float speed, float damping,
    vk::CommandBuffer command_buffer) {
    n_row = m;
    n_col = n;
    n_vertex = m * n;
    n_triangle = 2 * (m - 1) * (n - 1);
    time_step = dt;
    spatial_step = dx;

    float d = damping * dt + 2.0f;
    float e = (speed * speed) * (dt * dt) / (dx * dx);
    k1 = (damping * dt - 2.0f) / d;
    k2 = (4.0f - 8.0f * e) / d;
    k3 = (2.0f * e) / d;

    BuildImages(device, command_buffer);
    BuildLayouts(device);
}

void Wave::Update(float dt, vk::Pipeline pipeline, vk::CommandBuffer command_buffer) {
    static float t = 0.0f;
    t += dt;
    if (t >= time_step) {
        command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute_pipeline_layout.get(), 0,
            { compute_descriptor_set[curr_index] }, {});
        float k[] = { k1, k2, k3 };
        command_buffer.pushConstants(compute_pipeline_layout.get(), vk::ShaderStageFlagBits::eCompute,
            0, 3 * 4, k);

        uint32_t n_group_x = (n_col + 15) >> 4;
        uint32_t n_group_y = (n_row + 15) >> 4;
        command_buffer.dispatch(n_group_x, n_group_y, 1);

        curr_index = (curr_index + 1) % 3;
        curr_solution.swap(next_solution);
        prev_solution.swap(next_solution);

        t = 0.0f;
    }
}

void Wave::Disturb(int i, int j, float magnitude, vk::Pipeline pipeline, vk::CommandBuffer command_buffer) {
    assert(i > 1 && i < n_row - 2);
    assert(j > 1 && j < n_col - 2);

    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute_pipeline_layout.get(), 0,
        { compute_descriptor_set[curr_index] }, {});
    command_buffer.pushConstants(compute_pipeline_layout.get(), vk::ShaderStageFlagBits::eCompute,
        3 * 4, 1 * 4, &magnitude);
    int disturb_ind[] = { j, i };
    command_buffer.pushConstants(compute_pipeline_layout.get(), vk::ShaderStageFlagBits::eCompute,
        4 * 4, 2 * 4, disturb_ind);

    command_buffer.dispatch(1, 1, 1);
}

void Wave::PrepareDraw(vk::CommandBuffer command_buffer) {
    vk::ImageMemoryBarrier barrier(
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
        0, 0, curr_solution->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {}, { barrier });
}

void Wave::PrepareCompute(vk::CommandBuffer command_buffer) {
    vk::ImageMemoryBarrier barrier(
        vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral,
        0, 0, curr_solution->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {}, { barrier });
}

void Wave::BuildAndWriteDescriptorSets(const vku::Device *device, vk::DescriptorPool pool) {
    std::vector<vk::DescriptorSetLayout> layouts(3, compute_descriptor_set_layouts[0].get());
    vk::DescriptorSetAllocateInfo allocate_info(pool, layouts.size(), layouts.data());
    compute_descriptor_set = device->logical_device->allocateDescriptorSets(allocate_info);

    std::array<vk::DescriptorImageInfo, 3> image_infos = {
        vk::DescriptorImageInfo({}, prev_solution_view.get(), vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, curr_solution_view.get(), vk::ImageLayout::eGeneral),
        vk::DescriptorImageInfo({}, next_solution_view.get(), vk::ImageLayout::eGeneral),
    };
    std::array<vk::WriteDescriptorSet, 9> writes = {
        vk::WriteDescriptorSet(compute_descriptor_set[0], 0, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[0], nullptr, nullptr),
        vk::WriteDescriptorSet(compute_descriptor_set[0], 1, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[1], nullptr, nullptr),
        vk::WriteDescriptorSet(compute_descriptor_set[0], 2, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[2], nullptr, nullptr),

        vk::WriteDescriptorSet(compute_descriptor_set[1], 0, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[1], nullptr, nullptr),
        vk::WriteDescriptorSet(compute_descriptor_set[1], 1, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[2], nullptr, nullptr),
        vk::WriteDescriptorSet(compute_descriptor_set[1], 2, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[0], nullptr, nullptr),

        vk::WriteDescriptorSet(compute_descriptor_set[2], 0, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[2], nullptr, nullptr),
        vk::WriteDescriptorSet(compute_descriptor_set[2], 1, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[0], nullptr, nullptr),
        vk::WriteDescriptorSet(compute_descriptor_set[2], 2, 0, 1, vk::DescriptorType::eStorageImage,
            &image_infos[1], nullptr, nullptr),
    };

    device->logical_device->updateDescriptorSets(writes, {});
}

void Wave::BuildLayouts(const vku::Device *device) {
    compute_descriptor_set_layouts.resize(1);

    std::array<vk::DescriptorSetLayoutBinding, 3> img_binding = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute),
        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute),
    };
    vk::DescriptorSetLayoutCreateInfo img_create_info({}, img_binding.size(), img_binding.data());
    compute_descriptor_set_layouts[0] = device->logical_device->createDescriptorSetLayoutUnique(img_create_info);

    std::array<vk::PushConstantRange, 1> push_constants = {
        vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, 6 * 4)
    };

    auto layouts = vk::uniqueToRaw(compute_descriptor_set_layouts);
    vk::PipelineLayoutCreateInfo pipeline_layout_create_info({}, layouts.size(), layouts.data(),
        push_constants.size(), push_constants.data());
    compute_pipeline_layout = device->logical_device->createPipelineLayoutUnique(pipeline_layout_create_info);
}

void Wave::BuildImages(const vku::Device *device, vk::CommandBuffer command_buffer) {
    // create images
    prev_solution = std::make_unique<vku::Image>(device, vk::ImageType::e2D, vk::Format::eR32Sfloat,
        vk::Extent3D(n_col, n_row, 1), 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageLayout::eUndefined, vk::MemoryPropertyFlagBits::eDeviceLocal);

    curr_solution = std::make_unique<vku::Image>(device, vk::ImageType::e2D, vk::Format::eR32Sfloat,
        vk::Extent3D(n_col, n_row, 1), 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageLayout::eUndefined, vk::MemoryPropertyFlagBits::eDeviceLocal);

    next_solution = std::make_unique<vku::Image>(device, vk::ImageType::e2D, vk::Format::eR32Sfloat,
        vk::Extent3D(n_col, n_row, 1), 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageLayout::eUndefined, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // create image views
    vk::ImageViewCreateInfo view_create_info({}, prev_solution->image.get(), vk::ImageViewType::e2D,
        vk::Format::eR32Sfloat, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    prev_solution_view = device->logical_device->createImageViewUnique(view_create_info);

    view_create_info.setImage(curr_solution->image.get());
    curr_solution_view = device->logical_device->createImageViewUnique(view_create_info);

    view_create_info.setImage(next_solution->image.get());
    next_solution_view = device->logical_device->createImageViewUnique(view_create_info);

    // copy initial data
    std::vector<float> init_data(n_row * n_col, 0.0f);
    size_t data_size = init_data.size() * sizeof(float);
    staging_buffer = std::make_unique<vku::Buffer>(device, data_size, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void *mapped = device->logical_device->mapMemory(staging_buffer->device_memory.get(), 0, data_size);
    memcpy(mapped, init_data.data(), data_size);
    device->logical_device->unmapMemory(staging_buffer->device_memory.get());

    vk::ImageMemoryBarrier prev_barrier(
        {}, vk::AccessFlagBits::eMemoryWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        0, 0, prev_solution->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    vk::ImageMemoryBarrier curr_barrier(
        {}, vk::AccessFlagBits::eMemoryWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        0, 0, curr_solution->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, { prev_barrier, curr_barrier });

    vk::BufferImageCopy copy_region(0, 0, 0, { vk::ImageAspectFlagBits::eColor, 0, 0, 1 }, { 0, 0, 0 },
        { static_cast<uint32_t>(n_col), static_cast<uint32_t>(n_row), 1 });
    command_buffer.copyBufferToImage(staging_buffer->buffer.get(), prev_solution->image.get(),
        vk::ImageLayout::eTransferDstOptimal, { copy_region });
    command_buffer.copyBufferToImage(staging_buffer->buffer.get(), curr_solution->image.get(),
        vk::ImageLayout::eTransferDstOptimal, { copy_region });

    prev_barrier = vk::ImageMemoryBarrier(
        vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral,
        0, 0, prev_solution->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    curr_barrier = vk::ImageMemoryBarrier(
        vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        0, 0, curr_solution->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    vk::ImageMemoryBarrier next_barrier = vk::ImageMemoryBarrier(
        {}, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        0, 0, next_solution->image.get(), { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, { prev_barrier, next_barrier });
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, { curr_barrier });
}