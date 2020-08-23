add_executable(08_lighting
    main.cpp
    FrameResources.cpp
    Wave.cpp
)

target_link_libraries(08_lighting
    PRIVATE vk_common
)