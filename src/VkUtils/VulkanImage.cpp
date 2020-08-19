#include "VulkanImage.h"

namespace {

uint32_t FindMemoryType(vk::PhysicalDevice physical_device, uint32_t filter_mask,
                        vk::MemoryPropertyFlags properties) {
    auto memory_properties = physical_device.getMemoryProperties();
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((filter_mask & i) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("[FindMemoryType(Image)] Failed to find a suitable memory type");
}

}

namespace pepcy::vku {

Image::Image(const Device *device, vk::ImageType type, vk::Format format, const vk::Extent3D &extent,
             uint32_t mip_level, uint32_t array_layer, vk::SampleCountFlagBits sample_count, vk::ImageTiling tiling,
             vk::ImageUsageFlags usage, vk::ImageLayout layout, vk::MemoryPropertyFlags properties) {
    vk::ImageCreateInfo create_info({}, type, format, extent, mip_level, array_layer, sample_count, tiling, usage,
                                    vk::SharingMode::eExclusive, 0, nullptr, layout);
    image = device->logical_device->createImageUnique(create_info);

    auto memory_requirements = device->logical_device->getImageMemoryRequirements(image.get());
    vk::MemoryAllocateInfo allocate_info(memory_requirements.size,
                                         FindMemoryType(device->physical_device, memory_requirements.memoryTypeBits,
                                                        properties));
    device_memory = device->logical_device->allocateMemoryUnique(allocate_info);

    device->logical_device->bindImageMemory(image.get(), device_memory.get(), 0);
}

}