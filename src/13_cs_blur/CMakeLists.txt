add_executable(13_cs_blur
    main.cpp
    FrameResources.cpp
    Wave.cpp
    BlurFilter.cpp
)

target_link_libraries(13_cs_blur
    PRIVATE vk_common
)