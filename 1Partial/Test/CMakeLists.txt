cmake_minimum_required(VERSION 3.26)
project(Test CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(Test main.cu)

set_target_properties(Test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
