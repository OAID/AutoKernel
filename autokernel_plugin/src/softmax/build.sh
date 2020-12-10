g++ softmax_gen.cc ../../common/GenGen.cpp \
	-I /workspace/Halide/halide-build/include/ \
	-L /workspace/Halide/halide-build/src \
	-lHalide -std=c++11 -fno-rtti \
	-o softmax_gen

./softmax_gen -g halide_softmax -e c_header,assembly -o . target=host
