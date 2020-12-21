# Cuda matmul
This example shows how to use Halide to run code on a GPU(cuda). Source codes are from [Halide project]((https://github.com/halide/Halide)).


## 1. using nvidia-docker
we provide a cuda-docker image `openaialb/autokernel:cuda` with Halide and Tengine installed in it. To launch it, using nvidia-docker:
```
nvidia-docker pull openailab/autokernel:cuda
nvidia-docker run -it openailab/autokernel:cuda /bin/bash
```
How to install `nvidia-docker`? Please ref to [nvidia-docker install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian).
## 2. build
```
cd Autokernel/examples/cuda_matmul
mkdir build
cd build
cmake ..
make
```

## 3. test
```
./matmul_run
```
This test matmul of size M=N=K=512. Here's the performance on marchine with 2 GeForce GTX 1080 Ti.
```
autokernel time:        0.000488
cublas time:    0.000598
Success!
```
