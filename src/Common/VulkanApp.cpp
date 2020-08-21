#include "VulkanApp.h"

#include <iostream>
#include <unordered_set>

#define DEBUG

namespace {

void FrameBufferResizeCallback(GLFWwindow *window, int width, int height) {
    auto app = reinterpret_cast<VulkanApp *>(glfwGetWindowUserPointer(window));
    if (width != app->Width() || height != app->Height()) {
        app->SignalResize();
    }
}
void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto app = reinterpret_cast<VulkanApp *>(glfwGetWindowUserPointer(window));
    app->OnKey(key, action);
}
void CursorPosCallback(GLFWwindow *window, double x, double y) {
    auto app = reinterpret_cast<VulkanApp *>(glfwGetWindowUserPointer(window));
    uint32_t state = 0;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        state |= 1;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        state |= 2;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
        state |= 4;
    }
    app->OnMouse(x, y, state);
}

std::vector<const char *> GetRequiredExtensions() {
    uint32_t n_glfw_extensions;
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&n_glfw_extensions);
    std::vector<const char *> extensions(glfw_extensions, glfw_extensions + n_glfw_extensions);
#if defined(DEBUG) || defined(_DEBUG)
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
    return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type, const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
    void* p_user_data) {
    std::cerr << "[validation layer] " << p_callback_data->pMessage << std::endl;
    return VK_FALSE;
}

enum PhysicalDeviceScore : uint32_t {
    ePhysicalDeviceUnusable = 0,
    ePhysicalDeviceDedicatedTransferQueue = 1 << 0,
    ePhysicalDeviceDedicatedComputeQueue = 1 << 1,
    ePhysicalDeviceSamplerAnisotropy = 1 << 2
};

uint32_t GetPhysicalDeviceScore(vk::PhysicalDevice physical_device, vk::SurfaceKHR surface,
    const std::vector<const char *> &required_extensions) {
    // extensions
    auto supported_extensions = physical_device.enumerateDeviceExtensionProperties();
    std::unordered_set<std::string> required_set(required_extensions.begin(), required_extensions.end());
    for (const auto &extension : supported_extensions) {
        required_set.erase(std::string(extension.extensionName));
    }
    if (!required_set.empty()) {
        return ePhysicalDeviceUnusable;
    }

    // swap chain
    auto surface_formats = physical_device.getSurfaceFormatsKHR(surface);
    auto present_modes = physical_device.getSurfacePresentModesKHR(surface);
    if (surface_formats.empty() || present_modes.empty()) {
        return ePhysicalDeviceUnusable;
    }

    uint32_t score = 0;

    // queue
    auto queue_families = physical_device.getQueueFamilyProperties();
    bool has_graphics = false;
    for (uint32_t i = 0; i < queue_families.size(); i++) {
        const auto &property = queue_families[i];
        if ((property.queueFlags & vk::QueueFlagBits::eGraphics) && physical_device.getSurfaceSupportKHR(i, surface)) {
            has_graphics = true;
        } else if (property.queueFlags & vk::QueueFlagBits::eCompute) {
            score |= ePhysicalDeviceDedicatedComputeQueue;
        } else if (property.queueFlags & vk::QueueFlagBits::eTransfer) {
            score |= ePhysicalDeviceDedicatedTransferQueue;
        }
    }
    if (!has_graphics) {
        return ePhysicalDeviceUnusable;
    }

    // features
    vk::PhysicalDeviceFeatures supported_features = physical_device.getFeatures();
    if (supported_features.samplerAnisotropy) {
        score |= ePhysicalDeviceSamplerAnisotropy;
    }

    return score;
}

}

VulkanApp::~VulkanApp() {
    if (glfw_window) {
        glfwDestroyWindow(glfw_window);
        glfwTerminate();
    }
}

void VulkanApp::Initialize() {
    InitializeWindow();
    InitializeVulkan();
}

void VulkanApp::MainLoop() {
    timer.Reset();

    while (!glfwWindowShouldClose(glfw_window)) {
        glfwPollEvents();

        timer.Tick();
        CalcFrameStats();
        Update();
        Draw();
    }
}

void VulkanApp::InitializeWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfw_window = glfwCreateWindow(client_width, client_height, window_title.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(glfw_window, this);
    glfwSetFramebufferSizeCallback(glfw_window, FrameBufferResizeCallback);
    glfwSetKeyCallback(glfw_window, KeyCallback);
    glfwSetCursorPosCallback(glfw_window, CursorPosCallback);
}

void VulkanApp::InitializeVulkan() {
    vk::ApplicationInfo app_info("learn_vulkan", VK_MAKE_VERSION(1, 0, 0), "No Engine",
        VK_MAKE_VERSION(1, 0, 0), VK_MAKE_VERSION(1, 0, 0));
    auto extensions = GetRequiredExtensions();
#if defined(DEBUG) || defined(_DEBUG)
    const std::vector<const char *> layers = { "VK_LAYER_KHRONOS_validation" };
    vk::DebugUtilsMessageSeverityFlagsEXT severity_flags = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning;
    vk::DebugUtilsMessageTypeFlagsEXT type_flags = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
    vk::DebugUtilsMessengerCreateInfoEXT debug_messenger_create_info({}, severity_flags, type_flags, DebugCallback);
    vk::InstanceCreateInfo instance_create_info({}, &app_info, layers.size(), layers.data(),
        extensions.size(), extensions.data());
    instance_create_info.setPNext(&debug_messenger_create_info);
    instance = vk::createInstanceUnique(instance_create_info);
    dldi.init(instance.get(), vkGetInstanceProcAddr);
    debug_messenger = instance->createDebugUtilsMessengerEXTUnique(debug_messenger_create_info, nullptr, dldi);
#else
    vk::InstanceCreateInfo instance_create_info({}, &app_info, 0, nullptr, extensions.size(), extensions.data());
    instance = vk::createInstanceUnique(instance_create_info);
    dldi.init(instance.get(), vkGetInstanceProcAddr);
#endif
    VkSurfaceKHR surface_temp;
    if (glfwCreateWindowSurface(instance.get(), glfw_window, nullptr, &surface_temp) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }
    surface = vk::UniqueSurfaceKHR(surface_temp, instance.get());

    CreateDevice();
    CreateSwapchain();
    CreateCommandObjects();
    CreateSyncObjects();
}

void VulkanApp::CreateDevice() {
    const std::vector<const char *> extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    auto physical_devices = instance->enumeratePhysicalDevices();
    uint32_t max_score = 0;
    vk::PhysicalDevice physical_device;
    for (const auto &temp_device : physical_devices) {
        uint32_t score = GetPhysicalDeviceScore(temp_device, surface.get(), extensions);
        if (score > max_score) {
            max_score = score;
            physical_device = temp_device;
        }
#if defined(DEBUG) || defined(_DEBUG)
        std::cerr << "physical device name: " << temp_device.getProperties().deviceName << std::endl;
        std::cerr << "                score: " << score << std::endl;
#endif
    }
    vk::PhysicalDeviceFeatures physical_device_features {};
    if (max_score & ePhysicalDeviceSamplerAnisotropy) {
        physical_device_features.setSamplerAnisotropy(VK_TRUE);
    }

    device = std::make_unique<vku::Device>(physical_device, extensions, physical_device_features);

    graphics_queue = device->GraphicsQueue();
    // compute_queue = device->ComputeQueue();
    // transfer_queue = device->TransferQueue();
}

void VulkanApp::CreateSwapchain() {
    swapchain = std::make_unique<vku::Swapchain>(device.get(), surface.get(),
        vk::ImageUsageFlagBits::eColorAttachment, client_width, client_height);
}

void VulkanApp::CreateCommandObjects() {
    uint32_t graphics_queue_index = device->queue_families.graphics.value();
    vk::CommandPoolCreateInfo pool_create_info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphics_queue_index);
    command_pool = device->logical_device->createCommandPoolUnique(pool_create_info);

    vk::CommandBufferAllocateInfo buffer_allocate_info(command_pool.get(), vk::CommandBufferLevel::ePrimary, 1);
    main_command_buffer = std::move(device->logical_device->allocateCommandBuffersUnique(buffer_allocate_info)[0]);
    buffer_allocate_info.setCommandBufferCount(n_inflight_frames);
    command_buffers = device->logical_device->allocateCommandBuffersUnique(buffer_allocate_info);
}

void VulkanApp::CreateSyncObjects() {
    vk::SemaphoreCreateInfo semaphore_create_info {};
    image_available_semaphores.resize(n_inflight_frames);
    render_finish_semaphores.resize(n_inflight_frames);
    for (size_t i = 0; i < n_inflight_frames; i++) {
        image_available_semaphores[i] = device->logical_device->createSemaphoreUnique(semaphore_create_info);
        render_finish_semaphores[i] = device->logical_device->createSemaphoreUnique(semaphore_create_info);
    }

    vk::FenceCreateInfo fence_create_info(vk::FenceCreateFlagBits::eSignaled);
    fences.resize(n_inflight_frames);
    for (size_t i = 0; i < n_inflight_frames; i++) {
        fences[i] = device->logical_device->createFenceUnique(fence_create_info);
    }

    swapchain_image_fences.resize(swapchain->n_image);
}

void VulkanApp::SignalResize() {
    resized = true;
}

void VulkanApp::OnResize() {
    int width_, height_;
    glfwGetFramebufferSize(glfw_window, &width_, &height_);
    while (width_ == 0 || height_ == 0) {
        glfwGetFramebufferSize(glfw_window, &width_, &height_);
        glfwWaitEvents();
    }

    device->logical_device->waitIdle();
    client_width = width_;
    client_height = height_;

    swapchain.reset(nullptr);
    CreateSwapchain();
}

void VulkanApp::OnKey(int key, int action) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(glfw_window, 1);
    }
}

void VulkanApp::CalcFrameStats() {
    static int frame_cnt = 0;
    static double time_elapsed = 0.0;

    ++frame_cnt;
    double delta = timer.TotalTime() - time_elapsed;
    if (delta >= 1.0) {
        double fps = frame_cnt / delta;
        double mspf = 1000.0 / frame_cnt;
        std::string fps_str = std::to_string(fps);
        std::string mspf_str = std::to_string(mspf);
        std::string text = window_title + "  fps:" + fps_str + "  mspf:" + mspf_str;
        glfwSetWindowTitle(glfw_window, text.c_str());
        frame_cnt = 0;
        time_elapsed += 1.0;
    }
}