HALIDE_BUILD_DIR=/workspace/Halide/halide-build

EXE_FILE=06_gemm_optimization

g++ ${EXE_FILE}.cpp \
 -I ${HALIDE_BUILD_DIR}/include/ \
 -L ${HALIDE_BUILD_DIR}/src/ -lHalide\
  -lpthread -ldl -std=c++11 -lopenblas\
  -o ${EXE_FILE} 

export LD_LIBRARY_PATH=${HALIDE_BUILD_DIR}/src

export OMP_NUM_THREADS=4
export HL_NUM_THREADS=4

if [ ! -n "$1" ]; then
    echo "Usage:./build.sh <step>(step=1,2,..,7)"
    echo "e.g. execute step3:./build.sh 3"
    exit
fi
STEP=$1
echo "step = " ${STEP}
./${EXE_FILE} ${STEP}
