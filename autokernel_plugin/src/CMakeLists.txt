
# include directory list
include_directories(${TENGINE_ROOT}/build/source)
include_directories(${TENGINE_ROOT}/source)

ENABLE_LANGUAGE(ASM)

set(DRIVER_TARGET autokernel)
FILE(GLOB_RECURSE DRIVER_SRCS "*.cpp" "*.c" "*.s")
ADD_LIBRARY(${DRIVER_TARGET} SHARED ${DRIVER_SRCS})
target_link_libraries(${DRIVER_TARGET} ${TENGINE_LIBRARY})
