add_executable(09_texturing
    main.cpp
    FrameResources.cpp
    Wave.cpp
)

target_link_libraries(09_texturing
    PRIVATE vk_common
)