#pragma once

#include "VulkanDevice.h"

namespace pepcy::vku {

struct Image {
    Image(const Device *device, vk::ImageType type, vk::Format format, const vk::Extent3D &extent, uint32_t mip_level,
          uint32_t array_layer, vk::SampleCountFlagBits sample_count, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
          vk::ImageLayout layout, vk::MemoryPropertyFlags properties, vk::ImageCreateFlags create_flag = {});

    vk::UniqueImage image;
    vk::UniqueDeviceMemory device_memory;
};

}