#pragma once

#include "VulkanDevice.h"

namespace pepcy::vku {

struct Buffer {
    Buffer(const Device *device, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties);

    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory device_memory;
};

}