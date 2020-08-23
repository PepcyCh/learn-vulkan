#pragma once

#include <string>
#include <unordered_map>

#include "VulkanBuffer.h"

using namespace pepcy;

class VulkanUtil {
public:
    static vk::UniqueShaderModule CreateShaderModule(const std::string &spv_file, vk::Device device);

    static std::unique_ptr<vku::Buffer> CreateDeviceLocalBuffer(vku::Device *device, vk::BufferUsageFlagBits usage,
        vk::DeviceSize size, const void *data, std::unique_ptr<vku::Buffer> &staging_buffer,
        vk::CommandBuffer command_buffer);
};

struct SubmeshGeometry {
    uint32_t n_index = 0;
    uint32_t first_index = 0;
    int vertex_offset = 0;
};

struct MeshGeometry {
    std::string name;

    std::vector<void *> vertex_data;
    std::vector<void *> index_data;

    std::unique_ptr<vku::Buffer> vertex_buffer;
    std::unique_ptr<vku::Buffer> index_buffer;

    std::unique_ptr<vku::Buffer> vertex_staging_buffer;
    std::unique_ptr<vku::Buffer> index_staging_buffer;

    uint32_t vertex_stride;
    uint32_t vb_size;
    vk::IndexType index_type = vk::IndexType::eUint16;
    uint32_t ib_size;

    std::unordered_map<std::string, SubmeshGeometry> draw_args;
};