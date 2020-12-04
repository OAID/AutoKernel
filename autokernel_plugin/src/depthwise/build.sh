g++ depthwise_gen.cc ../../common/GenGen.cpp \
	-I /workspace/Halide/halide-build/include/ \
	-L /workspace/Halide/halide-build/src \
	-lHalide -std=c++11 -fno-rtti \
	-o depthwise_gen

./depthwise_gen -g halide_depthwise -e c_header,assembly -o . target=host
