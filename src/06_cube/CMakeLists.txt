add_executable(06_cube
    main.cpp
    FrameResources.cpp
)

target_link_libraries(06_cube
    PRIVATE vk_common
)