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