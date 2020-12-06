g++ maxpool_gen.cc ../../common/GenGen.cpp \
	-I ${HALIDE_DIR}/include \
	-L ${HALIDE_DIR}/lib \
	-lHalide -std=c++11 -fno-rtti \
	-o maxpool_gen

./maxpool_gen -g halide_maxpool -e c_header,assembly -o . target=host

g++ avepool_gen.cc ../../common/GenGen.cpp \
	-I ${HALIDE_DIR}/include \
	-L ${HALIDE_DIR}/lib \
	-lHalide -std=c++11 -fno-rtti \
	-o avepool_gen

./avepool_gen -g halide_avepool -e c_header,assembly -o . target=host
