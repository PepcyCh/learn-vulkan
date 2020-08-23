#include "FrameResources.h"

FrameResources::FrameResources(const vku::Device *device, vk::DescriptorPool descriptor_pool,
    vk::DescriptorSetLayout obj_set_layout, vk::DescriptorSetLayout pass_set_layout, size_t n_object,
    size_t n_wave_vertices) {
    std::vector<vk::DescriptorSetLayout> layouts(n_object, obj_set_layout);
    vk::DescriptorSetAllocateInfo allocate_info(descriptor_pool, layouts.size(), layouts.data());
    obj_descriptor_set = device->logical_device->allocateDescriptorSets(allocate_info);

    allocate_info.setDescriptorSetCount(1).setPSetLayouts(&pass_set_layout);
    pass_descriptor_set = device->logical_device->allocateDescriptorSets(allocate_info)[0];

    obj_ub = std::make_unique<HostVisibleBuffer<ObjectUB>>(device, n_object, vk::BufferUsageFlagBits::eUniformBuffer);
    pass_ub = std::make_unique<HostVisibleBuffer<PassUB>>(device, 1, vk::BufferUsageFlagBits::eUniformBuffer);

    wave_vb = std::make_unique<HostVisibleBuffer<Vertex>>(device, n_wave_vertices,
        vk::BufferUsageFlagBits::eVertexBuffer);
}
