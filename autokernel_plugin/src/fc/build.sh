g++ fc_gen.cc ../../common/GenGen.cpp \
	-I ${HALIDE_DIR}/include \
	-L ${HALIDE_DIR}/lib \
	-lHalide -std=c++11 -fno-rtti \
	-o fc_gen

./fc_gen -g halide_fc -e c_header,assembly -o . target=host
