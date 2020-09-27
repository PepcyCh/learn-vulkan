add_executable(19_displacement_map
    main.cpp
    FrameResources.cpp
)

target_link_libraries(19_displacement_map
    PRIVATE vk_common
)