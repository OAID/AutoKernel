g++ fc_gen.cc ../../common/GenGen.cpp \
	-I /workspace/Halide/halide-build/include/ \
	-L /workspace/Halide/halide-build/src \
	-lHalide -std=c++11 -fno-rtti \
	-o fc_gen

./fc_gen -g halide_fc -e c_header,assembly -o . target=host
