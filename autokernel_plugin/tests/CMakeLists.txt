project(test)                                                    
                                                                        
include_directories(${TENGINE_ROOT}/include)
include_directories(./common)
link_directories(${TENGINE_ROOT}/build/install/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(LINK_LIBS tengine-lite)
set(CMAKE_EXE_LINKER_FLAGS "-rdynamic -ldl")

add_executable(test_conv test_conv.cpp)
target_link_libraries(test_conv ${LINK_LIBS})

add_executable(tm_classification tm_classification.cpp common/tengine_operations.cpp)
target_link_libraries (tm_classification ${LINK_LIBS})
