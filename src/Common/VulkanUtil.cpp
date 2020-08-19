#include "VulkanUtil.h"

#include <fstream>

vk::UniqueShaderModule VulkanUtil::CreateShaderModule(const std::string &spv_file, vk::Device device) {
    std::ifstream fin(spv_file, std::ios::binary | std::ios::ate);
    if (!fin) {
        throw std::runtime_error("failed to open file '" + spv_file + "'");
    }
    size_t file_size = fin.tellg();
    std::vector<char> bytes(file_size);
    fin.seekg(0);
    fin.read(bytes.data(), file_size);
    fin.close();

    vk::ShaderModuleCreateInfo create_info({}, file_size, reinterpret_cast<const uint32_t *>(bytes.data()));
    auto shader_module = device.createShaderModuleUnique(create_info);
    return shader_module;
}

std::unique_ptr<vku::Buffer> VulkanUtil::CreateDeviceLocalBuffer(vku::Device *device, vk::BufferUsageFlagBits usage,
    vk::DeviceSize size, const void *data, std::unique_ptr<vku::Buffer> &staging_buffer,
    vk::CommandBuffer command_buffer) {
    staging_buffer = std::make_unique<vku::Buffer>(device, size, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void *temp_data = device->logical_device->mapMemory(staging_buffer->device_memory.get(), 0, size);
    memcpy(temp_data, data, size);
    device->logical_device->unmapMemory(staging_buffer->device_memory.get());

    auto result_buffer = std::make_unique<vku::Buffer>(device, size, usage | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::BufferCopy copy_region(0, 0, size);
    command_buffer.copyBuffer(staging_buffer->buffer.get(), result_buffer->buffer.get(), { copy_region });

    return result_buffer;
}