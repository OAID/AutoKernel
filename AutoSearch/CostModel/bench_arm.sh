GXX_TOOLCHAIN=aarch64-linux-gnu-g++-8

BATCH=$1
SAMPLE=$2

SAMPLE_DIR=./samples/batch_$BATCH/$SAMPLE/.

rm -f -r bin
mkdir bin
cp -r header/. bin
cp -r $SAMPLE_DIR bin

${GXX_TOOLCHAIN} test.cpp \
-I ./bin \
-Wall -O3 ./bin/runtime.a ./bin/random_pipeline.a \
-ldl -lpthread \
-o bin/test

scp -r bin firefly@10.12.1.50:/home/firefly/chenxi

ssh firefly@10.12.1.50 /home/firefly/chenxi/bin/test