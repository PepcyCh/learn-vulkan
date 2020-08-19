#pragma once

#include "VulkanBuffer.h"

using namespace pepcy;

struct FrameResources {
    FrameResources(const vku::Device *device, vk::DescriptorPool descriptor_pool, vk::DescriptorSetLayout set_layout,
        vk::DeviceSize ub_size);

    vk::DescriptorSet descriptor_set;
    vku::Buffer uniform_buffer;
};


