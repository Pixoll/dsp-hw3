cmake_minimum_required(VERSION 3.20)
project(dsp_hw3)

set(CMAKE_CXX_STANDARD 23)

if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif ()

add_executable(
    dsp_hw3
    src/main.cpp
)

target_include_directories(dsp_hw3 PUBLIC include)
