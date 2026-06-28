cmake_minimum_required(VERSION 3.20)
project(dsp_hw3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # Ampere (RTX 30xx); ajustar segun la GPU

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif ()

# CImg lee los PNG del dataset a traves de libpng.
find_package(PNG REQUIRED)

add_executable(dsp_hw3
    src/main.cpp
    src/exp1.cu
)
target_include_directories(dsp_hw3 PUBLIC include)
target_compile_definitions(dsp_hw3 PRIVATE cimg_use_png)
target_link_libraries(dsp_hw3 PRIVATE PNG::PNG)

# Utilidad standalone para el test de viabilidad de VRAM (Paso 0).
add_executable(test_vram src/test_vram.cu)
target_include_directories(test_vram PUBLIC include)
