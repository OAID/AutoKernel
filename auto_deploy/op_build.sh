
HALIDE_SOURCE_DIR=/workspace/Halide/
HALIDE_BUILD_DIR=/workspace/Halide/halide-build

g++ op_gen.cpp \
    ${HALIDE_SOURCE_DIR}/tools/GenGen.cpp \
    -o op.gen \
    -I ${HALIDE_BUILD_DIR}/include \
    -Wl,-rpath,${HALIDE_BUILD_DIR}/src \
    ${HALIDE_BUILD_DIR}/src/libHalide.so \
    -pthread -std=c++14  -ldl 

OUT_DIR=c_source

gen()
{
./op.gen \
-g $1  \
-o ${OUT_DIR} \
-e c_header,c_source \
target=x86-64-linux-no_runtime-no_bounds_query-no_asserts
}

gen halide_conv
gen halide_matmul
gen halide_relu
gen halide_maxpool
