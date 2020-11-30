## 如何快速使用AutoKernel Plugin

我们提供了AutoKernel的docker镜像，以便开发者可以快速搭建开发环境。
```
# 拉取镜像(可能需要一段时间，请耐心等待)
docker pull openailab/autokernel
# 启动容器，进入开发环境
docker run -it openailab/autokernel /bin/bash
```
docker里面提供了安装好的Halide和Tengine
```
/workspace/Halide	# Halide
/workspace/Tengine  # Tengine
```

克隆AutoKernel项目
```
git clone https://github.com/OAID/AutoKernel.git
```
我们首先看看`autokernel_plugin/src/`的文件目录：
```
autokernel_plugin/src/
|-- CMakeLists.txt
|-- direct_conv
|   |-- build.sh
|   |-- direct_conv.cpp
|   |-- direct_conv.h
|   |-- direct_conv_gen.cc
|-- im2col_conv
|   |-- build.sh
|   |-- im2col_conv.cpp
|   |-- im2col_conv.h
|   `-- im2col_conv_gen.cc
`-- plugin_init.cpp
```
可以看到`src`目录下有两个文件夹，每个文件夹的目录下有：
- xxx_gen.cc, 用Halide语言的算子描述(algorithm)和调度策略（schedule)
- build.sh 用于编译xxx_gen
- xxx.h 和 xxx.cpp是用Tengine算子接口封装的算子实现

一键生成算子汇编代码
```
cd AutoKernel/autokernel_plugin
chmod +x -R .
./scripts/generate.sh  #自动生成算子汇编文件
```
运行完这一步，可以看到原来的目录下多了两个自动生成的文件：
```bash
|-- im2col_conv
|   |-- halide_im2col_conv.h
|   |-- halide_im2col_conv.s
|-- direct_conv
|   |-- halide_direct_conv.h
|   `-- halide_direct_conv.s
```
接下来使用自动生成的文件，把Autokernel注册进tengine，一键编译 `libAutoKernel.so`
```
mkdir build
cd build
cmake ..
make -j4
```
生成的库在`/workspace/AutoKernel/autokernel_plugin/build/src/libautokernel.so`
运行测试，在测试代码中调用`load_tengine_plugin()`:
```
cd AutoKernel/autokernel_plugin
./build/tests/tm_classification -n squeezenet
```
分类网络的运行结果如下：

```
AutoKernel plugin inited
function:autokernel_plugin_init executed

...

Repeat 1 times, avg time per run is 55.932 ms
max time is 55.932 ms, min time is 55.932 ms
--------------------------------------
0.2732 - "n02123045 tabby, tabby cat"
0.2676 - "n02123159 tiger cat"
0.1810 - "n02119789 kit fox, Vulpes macrotis"
0.0818 - "n02124075 Egyptian cat"
0.0724 - "n02085620 Chihuahua"
--------------------------------------
ALL TEST DONE
```
可以看到，输出结果显示调用了`AutoKernel plugin`里的函数。