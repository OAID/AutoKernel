export HALIDE_DIR=/workspace/Halide/halide-build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HALIDE_DIR}/lib
for dir in `ls src`
do
    if [ -d src/$dir ] 
    then
	echo src/$dir
	cd src/$dir
	chmod +x build.sh
	./build.sh
        cd ../../
    fi
done 
