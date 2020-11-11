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
