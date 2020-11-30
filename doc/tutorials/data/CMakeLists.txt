project(tengine_tutorial)        
cmake_minimum_required(VERSION 3.10)

# Tengine path
set(TENGINE_ROOT /workspace/Tengine)   

include_directories(./common)
include_directories(${TENGINE_ROOT}/include)
link_directories(${TENGINE_ROOT}/build/install/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(LINK_LIBS tengine-lite)
set(CMAKE_EXE_LINKER_FLAGS "-rdynamic -ldl")

add_executable(02_tengine_tutorial 02_tengine_tutorial.cpp common/tengine_operations.c)
target_link_libraries (02_tengine_tutorial ${LINK_LIBS})
