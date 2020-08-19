#pragma once

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

#include "VulkanSwapchain.h"
#include "Timer.h"

using namespace pepcy;

class VulkanApp {
public:
    VulkanApp() = default;
    VulkanApp(const VulkanApp &rhs) = delete;
    VulkanApp &operator=(const VulkanApp &rhs) = delete;
    ~VulkanApp();

    uint32_t Width() const {
        return client_width;
    }
    uint32_t Height() const {
        return client_height;
    }
    float Aspect() const {
        return static_cast<float>(client_width) / client_height;
    }

    virtual void Initialize();
    void MainLoop();

    void SignalResize();

    virtual void OnKey(int key, int action);
    virtual void OnMouse(double x, double y, uint32_t state) = 0;

protected:
    virtual void Update() = 0;
    virtual void Draw() = 0;

    virtual void OnResize();

    void InitializeWindow();
    void InitializeVulkan();

    void CreateDevice();
    void CreateSwapchain();
    void CreateCommandObjects();
    void CreateSyncObjects();

    GLFWwindow *glfw_window = nullptr;
    uint32_t client_width = 2400;
    uint32_t client_height = 1350;
    bool resized = false;

    vk::UniqueInstance instance;
    vk::DispatchLoaderDynamic dldi;
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debug_messenger;
    vk::UniqueSurfaceKHR surface;

    std::unique_ptr<vku::Device> device;
    std::unique_ptr<vku::Swapchain> swapchain;
    uint32_t curr_frame = 0;

    vk::Queue graphics_queue;
    // vk::Queue compute_queue;
    // vk::Queue transfer_queue;

    vk::UniqueCommandPool main_command_pool;
    std::vector<vk::UniqueCommandPool> command_pools;
    vk::UniqueCommandBuffer main_command_buffer;
    std::vector<vk::UniqueCommandBuffer> command_buffers;

    std::vector<vk::UniqueSemaphore> image_available_semaphores;
    std::vector<vk::UniqueSemaphore> render_finish_semaphores;
    std::vector<vk::UniqueFence> fences;
    std::vector<vk::Fence> swapchain_image_fences;

    Timer timer;
};


