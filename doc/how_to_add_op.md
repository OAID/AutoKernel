## 如何快速开发一个自动优化的新算子

运用AutoKernel开发一个Tengine框架可用的算子的具体步骤分为以下两步：
1. 生成：编写算法描述和调度策略，生成目标后端的优化算子代码
   
2. 部署：将生成的优化算子代码通过plugin的形式集成进Tengine框架

--------------------------
本教程将以Relu算子为例，演示如何快速开发Tengine可用的自动优化算子。

![add_op.png](add_op.png)
### 1.执行`register_op.sh`，自动生成模板文件
我们提供了一个快速生成算子的脚本文件，根据模板生成这两个步骤需要的源文件和编译脚本。
```
cd AutoKernel/autokernel_plugin
chmod +x -R . 
./scripts/register_op.sh
```
根据提示填入：
```
op_name: relu
op_type: OP_RELU
```
可得到文件目录如下：
```
src/relu/relu.cpp
src/relu/relu.h
src/relu/relu_gen.cc
src/relu/build/sh
```
### 2.生成：编辑生成文件`relu_gen.cc`
该文件用于生成算子汇编代码。使用Halide语言描述算子的计算过程和调度策略schedule。
该示例中，schedule默认为空。

```
class halide_relu:public Halide::Generator<halide_relu>{
public:
    // args
    Input<Buffer<float>> input{"input", 4};
    Input<int> param{"param"};

    Output<Buffer<float>> output{"output", 4};

    void generate()
    {
        /* THE ALGORITHM */
        Var w("w"), h("h"), c("c"), n("n");
        Func halide_relu("halide_relu");
        halide_relu(w, h, c, n) = input(w, h, c, n);

        output(w, h, c, n) = select(param >= 0, max(param, halide_relu(w, h, c, n)), halide_relu(w, h, c, n));
    }

    void schedule()
    {
        /* THE SCHEDULE */
    }
};

```
### 3.部署：编辑`auto_relu.cpp`,一键编译生成`AutoKernel.so`

```
./scripts/generate.sh	# 一键生成所有算子所需的.s .h文件
mkdir build
cd build
cmake ..
make -j4
```

### 4.测试

测试用例仅供参考

```
#include "HalideBuffer.h"
#include <iostream>
#include "halide_relu.h"

int main(int argc, char **argv)
{
    int C = 1, W = 4, H = 4, N = 1;
    Halide::Runtime::Buffer<float> input_tensor(nullptr, W, H, C, N);
    Halide::Runtime::Buffer<float> output_tensor(nullptr, W, H, C, N);
    input_tensor.allocate();
    output_tensor.allocate();
    input_tensor.for_each_value([](float &x) {
        x = 2.0 * rand() / RAND_MAX - 1.0;
    });

    output_tensor.for_each_value([](float &x) {
        x = 2.0 * rand() / RAND_MAX - 1.0;
    });

    halide_relu(input_tensor, 0, output_tensor);

    printf("input:\n");
    for (int c = 0; c < input_tensor.dim(3).extent(); c++) {
        for (int z = 0; z < input_tensor.channels(); z++) {
            for (int y = 0; y < input_tensor.height(); y++) {
                for (int x = 0; x < input_tensor.width(); x++) {
                    std::cout<<input_tensor(x,y,z,0)<<" ";
                }
                std::cout<<"\n";
            }
            std::cout<<"\n";
        }
    }
    
    printf("output:\n");
    for (int c = 0; c < output_tensor.dim(3).extent(); c++) {
        for (int z = 0; z < output_tensor.channels(); z++) {
            for (int y = 0; y < output_tensor.height(); y++) {
                for (int x = 0; x < output_tensor.width(); x++) {
                    std::cout<<output_tensor(x,y,z,0)<<" ";
                }
                std::cout<<"\n";
            }
            std::cout<<"\n";
        }
    }

    return 0;
}
```
将该测试代码`test_relu.cpp`移至` AutoKernel/autokernel_plugin/build/`目录下后，通过如下命令行编译测试用例:

```
g++ test_relu.cpp ../src/relu/halide_relu.s -I ../include/ -I ../src/relu/ -std=c++11 -lpthread -ldl -O3 -o relu_run
```
运行获得测试结果
```
./relu_run
input:
0.680375 -0.211234 0.566198 0.59688 
0.823295 -0.604897 -0.329554 0.53645
-0.444451 0.10794 -0.0452059 0.25774
-0.270431 0.0268018 0.904459 0.83239

output:
0.680375 0 0.566198 0.59688 
0.823295 0 0 0.536459 
0 0.10794 0 0.257742 
0 0.0268018 0.904459 0.83239 
```

