cmake_minimum_required(VERSION 3.26)
project(TestCode CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(TestCode main.cu)

set_target_properties(TestCode PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
