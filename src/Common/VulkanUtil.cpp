#include "VulkanUtil.h"

#include <fstream>

#include "gli/gli.hpp"

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

std::unique_ptr<vku::Buffer> VulkanUtil::CreateDeviceLocalBuffer(const vku::Device *device,
    vk::BufferUsageFlagBits usage, vk::DeviceSize size, const void *data,
    std::unique_ptr<vku::Buffer> &staging_buffer, vk::CommandBuffer command_buffer) {
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

std::unique_ptr<Texture> VulkanUtil::LoadTextureFromFile(const vku::Device *device, const std::string &filename,
    vk::CommandBuffer command_buffer) {
    auto tex = std::make_unique<Texture>();
    tex->filename = filename;
    gli::texture gli_tex = gli::load(filename);

    vk::ImageType image_type = vk::ImageType::e1D;
    vk::ImageViewType image_view_type = vk::ImageViewType::e1D;
    auto gli_extent = gli_tex.extent(0);
    uint32_t layers = gli_tex.layers();
    bool is_cube = false;
    if (gli_extent.z == 1) {
        if (gli_extent.y == 1) {
            image_type = vk::ImageType::e1D;
            image_view_type = (gli_tex.layers() == 1 ? vk::ImageViewType::e1D : vk::ImageViewType::e1DArray);
        } else {
            image_type = vk::ImageType::e2D;
            if (gli_tex.faces() == 1) {
                image_view_type = (gli_tex.layers() == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray);
            } else {
                image_view_type = vk::ImageViewType::e2DArray;
                layers = 6;
                is_cube = true;
            }
        }
    } else {
        image_type = vk::ImageType::e3D;
        image_view_type = vk::ImageViewType::e3D;
    }
    vk::Extent3D extent(gli_extent.x, gli_extent.y, gli_extent.z);

    auto gli_format = gli_tex.format();
    auto format = static_cast<vk::Format>(gli_format);

    tex->image = std::make_unique<vku::Image>(device, image_type, format, extent, gli_tex.levels(), layers,
        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst, vk::ImageLayout::eUndefined,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::ImageViewCreateInfo image_view_create_info({}, tex->image->image.get(), image_view_type, format, {},
        { vk::ImageAspectFlagBits::eColor, 0, static_cast<uint32_t>(gli_tex.levels()), 0, layers });
    tex->image_view = device->logical_device->createImageViewUnique(image_view_create_info);

    tex->staging = std::make_unique<vku::Buffer>(device, gli_tex.size(), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void *buffer_data = device->logical_device->mapMemory(tex->staging->device_memory.get(), 0, gli_tex.size());
    memcpy(buffer_data, gli_tex.data(), gli_tex.size());
    device->logical_device->unmapMemory(tex->staging->device_memory.get());

    // transit undefined -> transfer dst optimal
    vk::ImageMemoryBarrier barrier({}, vk::AccessFlagBits::eTransferWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        0, 0, tex->image->image.get(),
        { vk::ImageAspectFlagBits::eColor, 0, static_cast<uint32_t>(gli_tex.levels()), 0, layers });
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, { barrier });

    std::vector<vk::BufferImageCopy> regions;
    if (!is_cube) {
        for (uint32_t layer = 0; layer < layers; layer++) {
            for (uint32_t level = 0; level < gli_tex.levels(); level++) {
                size_t offset = reinterpret_cast<uint8_t *>(gli_tex.data(layer, 0, level)) -
                    reinterpret_cast<uint8_t *>(gli_tex.data());
                auto sub_ext = gli_tex.extent(level);
                regions.emplace_back(offset, 0, 0,
                    vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, level, layer, 1),
                    vk::Offset3D(0, 0, 0), vk::Extent3D(sub_ext.x, sub_ext.y, sub_ext.z));
            }
        }
    } else {
        for (uint32_t face = 0; face < 6; face++) {
            for (uint32_t level = 0; level < gli_tex.levels(); level++) {
                size_t offset = reinterpret_cast<uint8_t *>(gli_tex.data(0, face, level)) -
                    reinterpret_cast<uint8_t *>(gli_tex.data());
                auto sub_ext = gli_tex.extent(level);
                regions.emplace_back(offset, 0, 0,
                    vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, level, face, 1),
                    vk::Offset3D(0, 0, 0), vk::Extent3D(sub_ext.x, sub_ext.y, sub_ext.z));
            }
        }
    }
    command_buffer.copyBufferToImage(tex->staging->buffer.get(), tex->image->image.get(),
        vk::ImageLayout::eTransferDstOptimal, regions);

    // transit transfer dst optimal -> shader read only optimal
    barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead)
        .setOldLayout(vk::ImageLayout::eTransferDstOptimal).setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {}, { barrier });

    return tex;
}