#include "VulkanSwapchain.h"

namespace {

vk::SurfaceFormatKHR ChooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &surface_formats) {
    for (const auto &format : surface_formats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }
    return surface_formats[0];
}

vk::PresentModeKHR ChoosePresentMode(const std::vector<vk::PresentModeKHR> &present_modes) {
    for (const auto &mode : present_modes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
            return mode;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D ChooseExtant(const vk::SurfaceCapabilitiesKHR &capabilities, uint32_t width, uint32_t height) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        width = std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        height = std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return { width, height };
    }
}

}

namespace pepcy::vku {

Swapchain::Swapchain(const Device *device, vk::SurfaceKHR surface, vk::ImageUsageFlags usage,
    uint32_t width, uint32_t height) : depth_image(device, vk::ImageType::e2D, vk::Format::eD24UnormS8Uint,
        { width, height, 1 }, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageLayout::eUndefined,
        vk::MemoryPropertyFlagBits::eDeviceLocal) {
    auto capabilities = device->physical_device.getSurfaceCapabilitiesKHR(surface);
    auto surface_formats = device->physical_device.getSurfaceFormatsKHR(surface);
    auto present_modes = device->physical_device.getSurfacePresentModesKHR(surface);

    auto surface_format = ChooseSurfaceFormat(surface_formats);
    format = surface_format.format;
    color_space = surface_format.colorSpace;
    auto present_mode = ChoosePresentMode(present_modes);
    extent = ChooseExtant(capabilities, width, height);

    n_image = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && n_image > capabilities.maxImageCount) {
        n_image = capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR create_info({}, surface, n_image, format, color_space, extent, 1, usage,
        vk::SharingMode::eExclusive, 0, nullptr, capabilities.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque, present_mode);
    swapchain = device->logical_device->createSwapchainKHRUnique(create_info);
    images = device->logical_device->getSwapchainImagesKHR(swapchain.get());

    n_image = images.size();
    image_views.resize(n_image);
    for (size_t i = 0; i < n_image; i++) {
        vk::ImageViewCreateInfo view_create_info({}, images[i], vk::ImageViewType::e2D, format, {},
            { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        image_views[i] = device->logical_device->createImageViewUnique(view_create_info);
    }

    depth_format = vk::Format::eD24UnormS8Uint;
    vk::ImageViewCreateInfo depth_view_create_info({}, depth_image.image.get(), vk::ImageViewType::e2D,
        depth_format, {}, { vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 });
    depth_image_view = device->logical_device->createImageViewUnique(depth_view_create_info);
}

}