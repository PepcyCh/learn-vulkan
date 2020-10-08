add_executable(20_shadow_map
    main.cpp
    FrameResources.cpp
    DepthRenderTarget.cpp
)

target_link_libraries(20_shadow_map
    PRIVATE vk_common
)