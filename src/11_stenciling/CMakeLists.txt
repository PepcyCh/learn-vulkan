add_executable(11_stenciling
    main.cpp
    FrameResources.cpp
)

target_link_libraries(11_stenciling
    PRIVATE vk_common
)