<p align="center"><img width="30%" src="doc/logo.png" /></p>

# AutoKernel

## 简介

随着人工智能的普及，深度学习网络的不断涌现，为了让各硬件(CPU, GPU, NPU,...)能够支持深度学习应用，各硬件芯片需要软件库去支持高性能的深度学习张量运算。目前，这些高性能计算库主要由资深HPC工程师(高性能计算优化工程师）进行开发，为了加快开发进程，缩短深度学习应用落地周期，自动化算子优化是一个趋势。

AutoKernel是由OPEN AI LAB提出的高性能算子自动优化工具，可以自动优化调度策略、生成底层优化代码，大幅减少各硬件芯片算子开发成本，提升算子优化效率，让工程师更快实现深度学习算法在各硬件芯片上的高性能部署。

## AutoKernel特色

- 低门槛
- 简单易用
- 高效率
  

## AutoKernel架构

![AutoKernel 架构](doc/architecture.png)

AutoKernel分为三个模块：
* 算子生成器: 

  该模块使用了开源项目[Halide](https://github.com/halide/Halide)；Halide是业界广泛使用的自动代码生成项目，它首次提出将计算和调度分离。该模块的输入是和硬件无关的算子计算描述，输出是相应后端的优化汇编代码/目标文件；

* 自动搜索模块：
  
  该模块可以通过最优化算法/搜索算法/机器学习/强化学习搜索出相应后端的最优算子的调度策略参数（该模块仍在开发中）；
* 算子部署插件（ AutoKernel Plugin）：
  
  [Tengine](https://github.com/OAID/Tengine)是OPEN AILAB开源的深度学习推理框架，实现了AI算法在不同硬件的快速高效部署。该模块实现了将自动生成的优化算子代码以plugin的形式一键集成到[Tengine](https://github.com/OAID/Tengine)中，实现自动优化算子的一键部署；


## 快速使用 Quick Start

我们提供了AutoKernel的docker镜像，以便开发者可以快速搭建开发环境。
```
# 拉取镜像(可能需要一段时间，请耐心等待)
docker pull openailab/autokernel
# 启动容器，进入开发环境
docker run -it openailab/autokernel /bin/bash
```
docker里面提供了安装好的Halide和Tengine：
```
/workspace/Halide	# Halide
/workspace/Tengine  # Tengine
```

克隆AutoKernel项目：
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

一键生成算子汇编代码：
```
cd AutoKernel/autokernel_plugin
chmod +x -R .
./scripts/generate.sh  #自动生成算子汇编文件
```
运行完这一步，可以看到原来的目录下多了两个自动生成的文件：
```
|-- im2col_conv
|   |-- halide_im2col_conv.h
|   |-- halide_im2col_conv.s
|-- direct_conv
|   |-- halide_direct_conv.h
|   `-- halide_direct_conv.s
```
接下来使用自动生成的文件，把Autokernel注册进tengine，一键编译 `libAutoKernel.so`：
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
## Docker
我们提供了以下三个docker镜像，镜像内安装了Halide和Tengine, 方便开发者直接使用:
- cpu: `openailab/autokernel`
- cuda: `openailab/autokernel:cuda`
- opencl: `openailab/autokernel:opencl`

具体的Dockerfile见 Dockerfiles目录

[NOTE]:
使用cuda镜像需要用`nvidia-docker`, 安装指南见 [nvidia-docker install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian).
```
nvidia-docker pull openaialb/autokernel:cuda
nvidia-docker run -it openaialb/autokernel:cuda /bin/bash
```

## 开发者指南

* 如何快速开发一个自动优化的新算子：[doc/how_to_add_op.md](doc/how_to_add_op.md)
* AutoKernel教程：[doc/tutorials](doc/tutorials/readme.md)

## Roadmap
- [Road map](doc/roadmap.md)

## License

- [Apache 2.0](LICENSE)


## 技术讨论
- Github issues
- QQ 群: 829565581
- Email: Support@openailab.com
- 微信公众号: Tengine开发者社区