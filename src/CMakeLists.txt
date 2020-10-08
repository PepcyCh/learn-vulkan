configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.h
)

add_subdirectory(Common)
add_subdirectory(VkUtils)

add_subdirectory(06_cube)
add_subdirectory(07_shapes)
add_subdirectory(07_land_wave)
add_subdirectory(08_lighting)
add_subdirectory(09_texturing)
add_subdirectory(10_blending)
add_subdirectory(11_stenciling)
add_subdirectory(12_gs_sprite)
add_subdirectory(13_cs_wave)
add_subdirectory(13_cs_blur)
add_subdirectory(14_ts_basic)
add_subdirectory(14_ts_bezier)
add_subdirectory(15_camera)
add_subdirectory(16_instance)
add_subdirectory(17_pick)
add_subdirectory(18_cubemap)
add_subdirectory(19_normal_map)
add_subdirectory(19_displacement_map)
add_subdirectory(20_shadow_map)