add_executable(16_instance
    main.cpp
    FrameResources.cpp
)

target_link_libraries(16_instance
    PRIVATE vk_common
)