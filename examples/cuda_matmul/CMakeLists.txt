cmake_minimum_required(VERSION 3.16)
project(cuda_mat_mul)

# This just checks whether CUDA is available ahead of time to allow CTest to
# skip this app when CUDA/cuBLAS are not installed on the system.
find_package(CUDA)
if (NOT CUDA_FOUND)
    message(FATAL_ERROR "[SKIP] Could NOT find CUDA")
endif ()

if (NOT CUDA_CUBLAS_LIBRARIES)
    message(FATAL_ERROR "[SKIP] Could NOT find cuBLAS")
endif ()

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_executable(matmul_gen matmul_gen.cpp)
target_link_libraries(matmul_gen PRIVATE Halide::Generator)

# Filters
add_halide_library(mat_mul FROM matmul_gen
                   TARGETS host
                   FEATURES cuda cuda_capability_50)

# Main executable
add_executable(matmul_run matmul_run.cpp)
target_include_directories(matmul_run PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(matmul_run PRIVATE Halide::Tools mat_mul ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

