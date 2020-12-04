# Halide初体验

在深入了解Halide之前，我们先来体验一下Halide的黑魔法。

进入AutoKernel的docker, docker里已有Halide的python环境，直接运行
```
python data/03_halide_magic.py
```
可以得到输出
```
func_origin__ cost 0.510215 second
func_parallel cost 0.122265 second
```
以上这个脚本执行了一个简单的函数计算：`func[x,y] = x + 10*y`
对比了两个函数的运行时间：
- func_origin: 默认函数
- func_parallel: 添加了Halide的一个调度策略：`func.parallel(y,4)`, 对y维度进行并行化，并行度为4

结果可以看到，第二个函数的耗时是第一个函数的四分之一。

<center>这，就是Halide的魔法！</center>

无需底层优化汇编知识，只需添加一行代码，就能得到比较好的优化效果


## Halide语言基础
要想调用Halide的调度策略，首先要掌握基本的Halide语言，用Halide语言来描述算子的计算。下面以简单的函数来演示Halide语言的基本数据结构。

- `变量 Var`：可以理解为函数的自变量，比如要描述一个图像的像素，需要两个变量x和y来描述 w维度和h维度的坐标。
- `函数 Func`：和数学上的函数类似，定义了一个计算过程。复杂的计算过程可以拆成多个小函数来实现。

### 示例一
本例子的函数计算公式为：`func(x,y)= 10*y + x`
用Halide语言来描述这个函数：
* Python:
    ```python
    import halide as hl

    x, y = hl.Var("x"), hl.Var("y")
    func = hl.Func("func")
    func[x,y] = x + 10*y
    ```
* C++
    ```c++
    #include "Halide.h"
    using namespace Halide;

    Var x("x"), y("y");
    Func func("func");

    func(x, y) = x + 10 * y;
    ```
Func的realize会计算函数在定义域的值并返回数值结果。调用了realize，函数才被即时编译(jit-compile),在这之前只是定义了函数的计算过程。

查看计算结果

* Python:
    ```python
    out = func.realize(3, 4)  # width, height = 3,4

    for j in range(out.height()):
        for i in range(out.width()):
            print("out[x=%i,y=%i]=%i"%(i,j,out[i,j]))
    ```
* C++
    ```c++
    Buffer<int32_t> out = func.realize(3, 4);
 
    for (int j = 0; j < out.height(); j++) {
        for (int i = 0; i < out.width(); i++) {
            printf("out[x=%d,y=%d]=%d",i,j,out(i,j));
            }
        }
    ```
这个函数的计算是：
```
                    wide = 3
                  x=0 x=1 x=2
                ------------
            y=0 |  0   1   2
hight = 4   y=1 | 10  11  12
            y=2 | 20  21  22
            y=3 | 30  31  32
```

完整的代码在[data/03_halide_basic.py](data/03_halide_basic.py)
可以直接运行：
```
python data/03_halide_basic.py
```
另外可以调用`func.trace_stores()`来跟踪函数的值

### 示例二
本示例演示如何喂入输入数据，取出输出数据
完整的代码在[data/03_halide_feed_data.py](data/03_halide_feed_data.py)

本示例的函数：
```
B(x,y)=A(x,y)+1
```
A是输入数据，可以定义Halide.Buffer,然后把numpy的array数据喂入buffer
```python
    # feed input
    input_data = np.ones((4,4),dtype=np.uint8)
    A = hl.Buffer(input_data)
```
定义函数B
```python
    i,j = hl.Var("i"), hl.Var("j")
    B = hl.Func("B")
    B[i,j] = A[i,j] + 1
```
获取输出数据, 有以下几种方式
```python
    # 1
        output = B.realize(4,4)
        print("out: \n",np.asanyarray(output))
    # 2
        output = hl.Buffer(hl.UInt(8),[4,4])
        B.realize(output)
        print("out: \n",np.asanyarray(output))
    # 3
        output_data = np.empty(input_data.shape, dtype=input_data.dtype,order="F")
        output = hl.Buffer(output_data)
        B.realize(output)
        print("out: \n",output_data)

```
可以直接运行完整代码：
```
python data/03_halide_feed_data.py
```