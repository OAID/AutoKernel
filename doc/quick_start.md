## 如何快速使用AutoKernel Plugin

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
一键生成算子汇编代码
```
cd AutoKernel/autokernel_plugin
chmod +x -R .
./script/generate.sh  #自动生成算子汇编文件
```
一键编译 `libAutoKernel.so`
```
mkdir build
cd build
cmake ..
make -j4
```
运行测试
```
cd AutoKernel/autokernel_plugin
./build/tests/tm_classification -n squeezenet
```
运行结果：

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