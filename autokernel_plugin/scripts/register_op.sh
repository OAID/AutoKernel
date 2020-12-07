#/usr/bin
#./register_op.sh op_dir_name op_func_name op_name
op_name=""
op_define_name=""
op_src_file=""
op_header_file=""
op_func_name=""

if [ ! -n "$1" ];then
    echo "please input op_name"
    read op_name
    echo "op_name:"$op_name
    echo "please input op_type, [eg.:OP_CONV, OP_POOL, ref @ tengine_op.h]"
    read op_define_name
    echo "op_type:"$op_define_name
    op_dir=src/$op_name
    op_src_file=$op_dir/$op_name.cpp
    op_header_file=$op_dir/$op_name.h
    op_define_name=${op_define_name^^}
    op_func_name=halide_${op_name}
else
    op_name=$1
    op_dir=src/$op_name
    op_define_name=${2^^}
    op_src_file=$op_dir/$op_name.cpp
    op_header_file=$op_dir/$op_name.h
    op_func_name=halide_${op_name}
fi


echo "op name is $op_name"
if [ ! -d $op_dir  ];then
    mkdir $op_dir
else
    rm -rf $op_dir
    mkdir $op_dir
fi

cp template/template.cpp $op_src_file
cp template/template.h $op_header_file
# cp generator/$op_name/$op_func_name.h $op_dir
# cp generator/$op_name/$op_func_name.s $op_dir

sed -i s/'template'/$op_name/g $op_src_file

sed -i s/'AutoKernel_Func'/$op_func_name/g $op_header_file
sed -i s/'AutoKernel_Func'/$op_func_name/g $op_src_file

sed -i s/'OP_CONV'/$op_define_name/g $op_src_file

sed -i s/'RegisterAutoKernelOP'/'RegisterAutoKernel'${op_name^}/g $op_header_file
sed -i s/'RegisterAutoKernelOP'/'RegisterAutoKernel'${op_name^}/g $op_src_file

# plugin_init.cpp
if [ `grep -c 'RegisterAutoKernel'${op_name^} src/plugin_init.cpp` -eq '0' ]; then
    line=`grep -n "autokernel_plugin_init"  src/plugin_init.cpp | cut -d ":" -f 1`
    sed -i '/register halide operator/a\    RegisterAutoKernel'${op_name^}'();' src/plugin_init.cpp
    sed -i '1a\#include "'${op_name}'/'${op_name}'.h"' src/plugin_init.cpp 
else
    echo "found"	
fi


# op_name_gen.cpp
op_gen_file=$op_dir/${op_name}_gen.cc
cp template/generator.cc $op_gen_file

sed -i s/'Halide_Func_Name'/$op_func_name/g $op_gen_file

# build.sh
cp template/build.sh $op_dir
sed -i s/'OP_NAME'/$op_name/g $op_dir/build.sh
chmod +x $op_dir/build.sh
