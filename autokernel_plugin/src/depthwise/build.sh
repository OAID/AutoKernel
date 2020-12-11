g++ depthwise_gen.cc ../../common/GenGen.cpp \
	-I ${HALIDE_DIR}/include \
	-L ${HALIDE_DIR}/lib \
        -lHalide -std=c++11 -fno-rtti \
	-o depthwise_gen

./depthwise_gen -g halide_depthwise -e c_header,assembly -o . target=host-no_runtime-no_asserts-no_bounds_query
