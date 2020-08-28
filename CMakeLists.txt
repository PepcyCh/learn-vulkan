cmake_minimum_required(VERSION 3.15)

project(learn-vulkan)

include(cmake/CPM.cmake)

find_package(Vulkan REQUIRED)

CPMAddPackage(
    NAME GLFW
    GITHUB_REPOSITORY glfw/glfw
    GIT_TAG 3.3.2
    OPTIONS
        "GLFW_BUILD_TESTS OFF"
        "GLFW_BUILD_EXAMPLES OFF"
        "GLFW_BULID_DOCS OFF"
)

CPMAddPackage(
    NAME Eigen
    VERSION 3.3.7
    URL https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    # Eigen's CMakelists are not intended for library use
    DOWNLOAD_ONLY YES 
)
if(Eigen_ADDED)
    add_library(Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()

CPMAddPackage(
    NAME stb
    GITHUB_REPOSITORY nothings/stb
    GIT_TAG master
    DOWNLOAD_ONLY YES
)
if(stb_ADDED)
    add_library(stb INTERFACE IMPORTED)
    target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})
endif()

CPMAddPackage(
    NAME gli
    GITHUB_REPOSITORY g-truc/gli
    GIT_TAG 0.8.2
    OPTIONS
        "GLI_TEST_ENABLE OFF"
)

CPMAddPackage(
    NAME glm
    GITHUB_REPOSITORY g-truc/glm
    GIT_TAG 0.9.8
    OPTIONS
        "GLM_TEST_ENABLE OFF"
)

add_subdirectory(src)