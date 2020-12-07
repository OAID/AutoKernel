g++ direct_conv_gen.cc ../../common/GenGen.cpp \
	-I /workspace/Halide/halide-build/include/ \
	-L /workspace/Halide/halide-build/src \
	-lHalide -std=c++11 -fno-rtti \
	-o direct_conv_gen

./direct_conv_gen -g halide_direct_conv -e c_header,assembly -o . target=host
