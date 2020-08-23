#pragma once

#include "VulkanBuffer.h"

using namespace pepcy;

template <typename T>
class HostVisibleBuffer {
public:
    HostVisibleBuffer(const vku::Device *device, size_t count, vk::BufferUsageFlags usage) :
        logical_device(device->logical_device.get()), count(count) {
        buffer = std::make_unique<vku::Buffer>(device, count * obj_size, usage,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        mapped_data = reinterpret_cast<T *>(logical_device.mapMemory(buffer->device_memory.get(), 0, count * obj_size));
    }
    ~HostVisibleBuffer() {
        if (mapped_data) {
            logical_device.unmapMemory(buffer->device_memory.get());
        }
    }

    vku::Buffer *Buffer() const {
        return buffer.get();
    }

    void CopyData(size_t index, const T &obj) {
        memcpy(mapped_data + index, &obj, obj_size);
    }

private:
    std::unique_ptr<vku::Buffer> buffer;
    T *mapped_data;
    inline static const size_t obj_size = sizeof(T);
    size_t count;
    vk::Device logical_device;
};


