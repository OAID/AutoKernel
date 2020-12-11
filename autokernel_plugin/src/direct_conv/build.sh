g++ direct_conv_gen.cc ../../common/GenGen.cpp \
	-I ${HALIDE_DIR}/include \
	-L ${HALIDE_DIR}/lib \
	-lHalide -std=c++11 -fno-rtti \
	-o direct_conv_gen

./direct_conv_gen -g halide_direct_conv -e c_header,assembly -o . target=host
