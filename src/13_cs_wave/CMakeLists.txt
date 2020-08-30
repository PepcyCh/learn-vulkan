add_executable(13_cs_wave
    main.cpp
    FrameResources.cpp
    Wave.cpp
)

target_link_libraries(13_cs_wave
    PRIVATE vk_common
)