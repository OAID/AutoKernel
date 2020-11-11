for dir in `ls src`
do
    if [ -d src/$dir ] 
    then
	echo src/$dir
	cd src/$dir
	rm *gen
	rm halide*
        cd ../../
    fi
done 
