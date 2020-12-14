g++ normalize_gen.cc ../../common/GenGen.cpp \
	-I ${HALIDE_DIR}/include \
	-L ${HALIDE_DIR}/lib \
	-lHalide -std=c++11 -fno-rtti \
	-o normalize_gen

./normalize_gen -g halide_normalize -e c_header,assembly -o . target=host
