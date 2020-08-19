#pragma once

#include "VulkanBuffer.h"

using namespace pepcy;

class VulkanUtil {
public:
    static vk::UniqueShaderModule CreateShaderModule(const std::string &spv_file, vk::Device device);

    static std::unique_ptr<vku::Buffer> CreateDeviceLocalBuffer(vku::Device *device, vk::BufferUsageFlagBits usage,
        vk::DeviceSize size, const void *data, std::unique_ptr<vku::Buffer> &staging_buffer,
        vk::CommandBuffer command_buffer);
};


