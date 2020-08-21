#include "VulkanDevice.h"

#include <cassert>

namespace pepcy::vku {

Device::Device(vk::PhysicalDevice physical_device_, const std::vector<const char *> &extensions,
    const vk::PhysicalDeviceFeatures &features) {
    assert(physical_device_ && "[vku::Device::Device] Create vku::Device with a null physical device");
    physical_device = physical_device_;

    auto queue_family_properties = physical_device.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queue_family_properties.size(); i++) {
        const auto &property = queue_family_properties[i];
        if (property.queueFlags & vk::QueueFlagBits::eGraphics) {
            queue_families.graphics = i;
        } else if (property.queueFlags & vk::QueueFlagBits::eCompute) {
            queue_families.compute = i;
        } else if (property.queueFlags & vk::QueueFlagBits::eTransfer) {
            queue_families.transfer = i;
        }
    }
    assert(queue_families.graphics.has_value() && "[vku::Device::Device] No graphics queue");

    std::vector<uint32_t> queue_family_indices = { queue_families.graphics.value() };
    if (queue_families.compute.has_value()) {
        queue_family_indices.push_back(queue_families.compute.value());
    }
    if (queue_families.transfer.has_value()) {
        queue_family_indices.push_back(queue_families.transfer.value());
    }

    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos(queue_family_indices.size());
    for (size_t i = 0; i < queue_family_indices.size(); i++) {
        float priority = 1.0f;
        queue_create_infos[i].setQueueFamilyIndex(queue_family_indices[i]).setQueueCount(1)
            .setPQueuePriorities(&priority);
    }

    vk::DeviceCreateInfo create_info({}, queue_create_infos.size(), queue_create_infos.data(),
        0, nullptr, extensions.size(), extensions.data(), &features);
    logical_device = physical_device.createDeviceUnique(create_info);
}

vk::Queue Device::GraphicsQueue() const {
    return logical_device->getQueue(queue_families.graphics.value(), 0);
}

vk::Queue Device::ComputeQueue() const {
    if (queue_families.compute.has_value()) {
        return logical_device->getQueue(queue_families.compute.value(), 0);
    } else {
        return logical_device->getQueue(queue_families.graphics.value(), 0);
    }
}

vk::Queue Device::TransferQueue() const {
    if (queue_families.transfer.has_value()) {
        return logical_device->getQueue(queue_families.transfer.value(), 0);
    } else {
        return logical_device->getQueue(queue_families.graphics.value(), 0);
    }
}

}