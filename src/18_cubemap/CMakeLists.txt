add_executable(18_cubemap
    main.cpp
    FrameResources.cpp
    CubemapRenderTarget.cpp
)

target_link_libraries(18_cubemap
    PRIVATE vk_common
)