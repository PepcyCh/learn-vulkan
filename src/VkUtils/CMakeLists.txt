add_library(vk_util
    VulkanBuffer.cpp
    VulkanDevice.cpp
    VulkanImage.cpp
    VulkanSwapchain.cpp
)

target_include_directories(vk_util
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(vk_util
    PUBLIC Vulkan::Vulkan
)

target_compile_features(vk_util PUBLIC cxx_std_17)