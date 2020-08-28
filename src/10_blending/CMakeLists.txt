add_executable(10_blending
    main.cpp
    FrameResources.cpp
    Wave.cpp
)

target_link_libraries(10_blending
    PRIVATE vk_common
)