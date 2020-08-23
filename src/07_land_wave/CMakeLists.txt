add_executable(07_land_wave
    main.cpp
    FrameResources.cpp
    Wave.cpp
)

target_link_libraries(07_land_wave
    PRIVATE vk_common
)