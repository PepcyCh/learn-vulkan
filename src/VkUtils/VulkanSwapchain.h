#pragma once

#include "VulkanImage.h"

namespace pepcy::vku {

struct Swapchain {
    Swapchain(const Device *device, vk::SurfaceKHR surface, vk::ImageUsageFlags usage,
              uint32_t width, uint32_t height);

    vk::UniqueSwapchainKHR swapchain;
    uint32_t n_image;
    vk::Extent2D extent;
    vk::Format format;
    vk::Format depth_format;
    vk::ColorSpaceKHR color_space;
    std::vector<vk::Image> images;
    std::vector<vk::UniqueImageView> image_views;
    Image depth_image;
    vk::UniqueImageView depth_image_view;
};

}