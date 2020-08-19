#include "FrameResources.h"

FrameResources::FrameResources(const vku::Device *device, vk::DescriptorPool descriptor_pool,
    vk::DescriptorSetLayout set_layout, vk::DeviceSize ub_size) :
    uniform_buffer(device, ub_size, vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent) {
    vk::DescriptorSetAllocateInfo allocate_info(descriptor_pool, 1, &set_layout);
    descriptor_set = device->logical_device->allocateDescriptorSets(allocate_info)[0];

    vk::DescriptorBufferInfo ub_info(uniform_buffer.buffer.get(), 0, ub_size);
    std::array<vk::WriteDescriptorSet, 1> writes = {
        vk::WriteDescriptorSet(descriptor_set, 0, 0, 1,vk::DescriptorType::eUniformBuffer, nullptr, &ub_info, nullptr)
    };
    device->logical_device->updateDescriptorSets(writes, {});
}
