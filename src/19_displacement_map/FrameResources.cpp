#include "FrameResources.h"

FrameResources::FrameResources(const vku::Device *device, vk::DescriptorPool descriptor_pool,
    vk::DescriptorSetLayout obj_set_layout, size_t n_object,
    vk::DescriptorSetLayout tex_set_layout, vk::DescriptorSetLayout cube_set_layout,
    vk::DescriptorSetLayout mat_set_layout, size_t n_mat,
    vk::DescriptorSetLayout pass_set_layout, size_t n_pass,
    vk::DescriptorSetLayout waves_set_layout, size_t n_wavs) {
    std::vector<vk::DescriptorSetLayout> layouts(n_object, obj_set_layout);
    vk::DescriptorSetAllocateInfo allocate_info(descriptor_pool, layouts.size(), layouts.data());
    obj_set = device->logical_device->allocateDescriptorSets(allocate_info);

    tex_set.resize(2);
    allocate_info.setDescriptorSetCount(1).setPSetLayouts(&tex_set_layout);
    tex_set[0] = device->logical_device->allocateDescriptorSets(allocate_info)[0];
    allocate_info.setDescriptorSetCount(1).setPSetLayouts(&cube_set_layout);
    tex_set[1] = device->logical_device->allocateDescriptorSets(allocate_info)[0];

    allocate_info.setDescriptorSetCount(1).setPSetLayouts(&mat_set_layout);
    mat_set = device->logical_device->allocateDescriptorSets(allocate_info);

    layouts.assign(n_pass, pass_set_layout);
    allocate_info.setDescriptorSetCount(n_pass).setPSetLayouts(layouts.data());
    pass_set = device->logical_device->allocateDescriptorSets(allocate_info);

    layouts.assign(n_wavs, waves_set_layout);
    allocate_info.setDescriptorSetCount(n_wavs).setPSetLayouts(layouts.data());
    waves_set = device->logical_device->allocateDescriptorSets(allocate_info);

    obj_ub = std::make_unique<HostVisibleBuffer<ObjectUB>>(device, n_object, vk::BufferUsageFlagBits::eUniformBuffer);
    mat_ub = std::make_unique<HostVisibleBuffer<MaterialData>>(device, n_mat, vk::BufferUsageFlagBits::eStorageBuffer);
    vert_pass_ub = std::make_unique<HostVisibleBuffer<VertPassUB>>(device, n_pass,
        vk::BufferUsageFlagBits::eUniformBuffer);
    frag_pass_ub = std::make_unique<HostVisibleBuffer<FragPassUB>>(device, n_pass,
        vk::BufferUsageFlagBits::eUniformBuffer);
    waves_ub = std::make_unique<HostVisibleBuffer<WavesTexTransform>>(device, n_wavs,
        vk::BufferUsageFlagBits::eUniformBuffer);
}
