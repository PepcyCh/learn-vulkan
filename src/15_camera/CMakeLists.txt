add_executable(15_camera
    main.cpp
    FrameResources.cpp
)

target_link_libraries(15_camera
    PRIVATE vk_common
)