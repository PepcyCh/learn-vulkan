#pragma once

#include "vulkan/vulkan.hpp"

namespace pepcy::vku {

struct  Device {
    Device(vk::PhysicalDevice physical_device_, const std::vector<const char *> &extensions,
        const vk::PhysicalDeviceFeatures &features);

    vk::Queue GraphicsQueue() const;
    vk::Queue ComputeQueue() const;
    vk::Queue TransferQueue() const;

    vk::PhysicalDevice physical_device;
    vk::UniqueDevice logical_device;

    struct {
        std::optional<uint32_t> graphics;
        std::optional<uint32_t> compute;
        std::optional<uint32_t> transfer;
    } queue_families;
};

}