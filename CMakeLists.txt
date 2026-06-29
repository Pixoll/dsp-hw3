cmake_minimum_required(VERSION 3.20)
project(dsp_hw3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86 89 120) # RTX 30xx 40xx 50xx series

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif ()

# CImg requirement for dataset
find_package(PNG REQUIRED)

add_executable(dsp_hw3
    src/main.cu
    src/experiment1.cu
    src/experiment2.cu
    src/preprocess.hpp
    src/args.hpp
    src/experiment2.h
)
target_include_directories(dsp_hw3 PUBLIC include)
target_compile_definitions(dsp_hw3 PRIVATE cimg_use_png)
target_link_libraries(dsp_hw3 PRIVATE PNG::PNG)

# check available GPU memory
add_executable(test_vram src/test_vram.cu)
target_include_directories(test_vram PUBLIC include)
