#include "FrameResources.h"

FrameResources::FrameResources(const vku::Device *device, vk::DescriptorPool descriptor_pool,
    vk::DescriptorSetLayout obj_set_layout, size_t n_object,
    vk::DescriptorSetLayout tex_set_layout, size_t n_tex,
    vk::DescriptorSetLayout mat_set_layout, size_t n_mat,
    vk::DescriptorSetLayout pass_set_layout, size_t n_pass) {
    vk::DescriptorSetAllocateInfo allocate_info(descriptor_pool, 1, &obj_set_layout);
    obj_set = device->logical_device->allocateDescriptorSets(allocate_info);

    allocate_info.setDescriptorSetCount(1).setPSetLayouts(&tex_set_layout);
    tex_set = device->logical_device->allocateDescriptorSets(allocate_info);

    allocate_info.setDescriptorSetCount(1).setPSetLayouts(&mat_set_layout);
    mat_set = device->logical_device->allocateDescriptorSets(allocate_info);

    std::vector<vk::DescriptorSetLayout> layouts(n_pass, pass_set_layout);
    allocate_info.setDescriptorSetCount(n_pass).setPSetLayouts(layouts.data());
    pass_set = device->logical_device->allocateDescriptorSets(allocate_info);

    obj_ub = std::make_unique<HostVisibleBuffer<InstanceData>>(device, n_object, vk::BufferUsageFlagBits::eStorageBuffer);
    mat_ub = std::make_unique<HostVisibleBuffer<MaterialData>>(device, n_mat, vk::BufferUsageFlagBits::eStorageBuffer);
    vert_pass_ub = std::make_unique<HostVisibleBuffer<VertPassUB>>(device, 1, vk::BufferUsageFlagBits::eUniformBuffer);
    frag_pass_ub = std::make_unique<HostVisibleBuffer<FragPassUB>>(device, 1, vk::BufferUsageFlagBits::eUniformBuffer);
}
