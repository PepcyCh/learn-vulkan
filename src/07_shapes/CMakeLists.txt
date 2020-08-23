add_executable(07_shapes
    main.cpp
    FrameResources.cpp
)

target_link_libraries(07_shapes
    PRIVATE vk_common
)