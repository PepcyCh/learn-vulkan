#include "FrameResources.h"

FrameResources::FrameResources(const vku::Device *device, vk::DescriptorPool descriptor_pool,
    vk::DescriptorSetLayout obj_set_layout, size_t n_object,
    vk::DescriptorSetLayout tex_set_layout, size_t n_tex,
    vk::DescriptorSetLayout mat_set_layout, size_t n_mat,
    vk::DescriptorSetLayout pass_set_layout, size_t n_pass,
    vk::DescriptorSetLayout disp_set_layout, size_t n_disp) {
    std::vector<vk::DescriptorSetLayout> layouts(n_object, obj_set_layout);
    vk::DescriptorSetAllocateInfo allocate_info(descriptor_pool, layouts.size(), layouts.data());
    obj_set = device->logical_device->allocateDescriptorSets(allocate_info);

    layouts.assign(n_tex, tex_set_layout);
    allocate_info.setDescriptorSetCount(n_tex).setPSetLayouts(layouts.data());
    tex_set = device->logical_device->allocateDescriptorSets(allocate_info);

    layouts.assign(n_mat, mat_set_layout);
    allocate_info.setDescriptorSetCount(n_mat).setPSetLayouts(layouts.data());
    mat_set = device->logical_device->allocateDescriptorSets(allocate_info);

    layouts.assign(n_pass, pass_set_layout);
    allocate_info.setDescriptorSetCount(n_pass).setPSetLayouts(layouts.data());
    pass_set = device->logical_device->allocateDescriptorSets(allocate_info);

    layouts.assign(n_disp, disp_set_layout);
    allocate_info.setDescriptorSetCount(n_disp).setPSetLayouts(layouts.data());
    disp_set = device->logical_device->allocateDescriptorSets(allocate_info);

    obj_ub = std::make_unique<HostVisibleBuffer<ObjectUB>>(device, n_object, vk::BufferUsageFlagBits::eUniformBuffer);
    mat_ub = std::make_unique<HostVisibleBuffer<MaterialUB>>(device, n_mat, vk::BufferUsageFlagBits::eUniformBuffer);
    vert_pass_ub = std::make_unique<HostVisibleBuffer<VertPassUB>>(device, 1, vk::BufferUsageFlagBits::eUniformBuffer);
    frag_pass_ub = std::make_unique<HostVisibleBuffer<FragPassUB>>(device, 1, vk::BufferUsageFlagBits::eUniformBuffer);
}
