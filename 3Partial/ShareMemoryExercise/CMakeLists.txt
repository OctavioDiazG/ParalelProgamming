cmake_minimum_required(VERSION 3.26)
project(ShareMemoryExercise CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(ShareMemoryExercise main.cu)

set_target_properties(ShareMemoryExercise PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
