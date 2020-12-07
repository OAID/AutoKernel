g++ OP_NAME_gen.cc ../../common/GenGen.cpp \
	-I /workspace/Halide/halide-build/include/ \
	-L /workspace/Halide/halide-build/src \
	-lHalide -std=c++11 -fno-rtti \
	-o OP_NAME_gen

./OP_NAME_gen -g halide_OP_NAME -e c_header,assembly -o . target=host
