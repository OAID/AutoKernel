g++ softmax_gen.cc ../../common/GenGen.cpp \
	-I ${HALIDE_DIR}/include \
	-L ${HALIDE_DIR}/lib \
	-lHalide -std=c++11 -fno-rtti \
	-o softmax_gen

./softmax_gen -g halide_softmax -e c_header,assembly -o . target=host
