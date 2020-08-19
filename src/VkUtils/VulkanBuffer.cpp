#include "VulkanBuffer.h"

namespace {

uint32_t FindMemoryType(vk::PhysicalDevice physical_device, uint32_t filter_mask,
                              vk::MemoryPropertyFlags properties) {
    auto memory_properties = physical_device.getMemoryProperties();
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((filter_mask & i) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("[FindMemoryType(Buffer)] Failed to find a suitable memory type");
}

}

namespace pepcy::vku {

Buffer::Buffer(const Device *device, vk::DeviceSize size, vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags properties) {
    vk::BufferCreateInfo create_info({}, size, usage);
    buffer = device->logical_device->createBufferUnique(create_info);

    auto memory_requirements = device->logical_device->getBufferMemoryRequirements(buffer.get());
    vk::MemoryAllocateInfo allocate_info(memory_requirements.size,
                                         FindMemoryType(device->physical_device, memory_requirements.memoryTypeBits,
                                                        properties));
    device_memory = device->logical_device->allocateMemoryUnique(allocate_info);

    device->logical_device->bindBufferMemory(buffer.get(), device_memory.get(), 0);
}

}