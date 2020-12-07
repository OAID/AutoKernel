g++ im2col_conv_gen.cc ../../common/GenGen.cpp \
	-I ${HALIDE_DIR}/include \
	-L ${HALIDE_DIR}/lib \
	-lHalide -std=c++11 -fno-rtti \
	-o im2col_conv_gen

./im2col_conv_gen -g halide_im2col_conv -e c_header,assembly -o . target=host-no_runtime-no_asserts-no_bounds_query
