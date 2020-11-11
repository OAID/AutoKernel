project(AutoKernel)

set(TENGINE_ROOT /workspace/Tengine)

if(NOT DEFINED TENGINE_ROOT)
    message(FATAL_ERROR "please set TENGINE_ROOT for tengine directory")
endif()

#set(TENGINE_LIBRARY ${TENGINE_ROOT}/build/install/lib/libtengine-lite.so)

cmake_minimum_required(VERSION 3.0)

add_definitions(-Wall)
add_definitions(-fPIC)
#add_definitions(-g)
add_definitions(-O3)
add_definitions(-funroll-loops)

include_directories(include/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-overloaded-virtual")
set(CMAKE_CXX_STANDARD 11)

# sub directories
add_subdirectory(src)
add_subdirectory(tests)
