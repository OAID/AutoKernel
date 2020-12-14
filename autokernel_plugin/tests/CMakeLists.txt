project(test)                                                    
                                                                        
include_directories(${TENGINE_DIR}/include)
include_directories(./common)
link_directories(${TENGINE_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(LINK_LIBS tengine-lite)
set(CMAKE_EXE_LINKER_FLAGS "-rdynamic -ldl")

add_executable(test_conv test_conv.cpp)
target_link_libraries(test_conv ${LINK_LIBS})

add_executable(test_depthwise test_depthwise.cpp)
target_link_libraries(test_depthwise ${LINK_LIBS})

add_executable(test_fc test_fc.cpp)
target_link_libraries(test_fc ${LINK_LIBS})

add_executable(test_pool test_pool.cpp)
target_link_libraries(test_pool ${LINK_LIBS})

add_executable(test_softmax test_softmax.cpp)
target_link_libraries(test_softmax ${LINK_LIBS})

add_executable(test_normalize test_normalize.cpp)
target_link_libraries(test_normalize ${LINK_LIBS})

add_executable(tm_classification tm_classification.cpp common/tengine_operations.cpp)
target_link_libraries (tm_classification ${LINK_LIBS})
