add_executable(12_gs_sprite
    main.cpp
    FrameResources.cpp
    Wave.cpp
)

target_link_libraries(12_gs_sprite
    PRIVATE vk_common
)