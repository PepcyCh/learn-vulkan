add_library(vk_common
    GeometryGenerator.cpp
    MathUtil.cpp
    Timer.cpp
    VulkanApp.cpp
    VulkanUtil.cpp
)

target_compile_features(vk_common PUBLIC cxx_std_17)

target_include_directories(vk_common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(vk_common
    PUBLIC Eigen vk_util glfw
)