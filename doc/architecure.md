# AutoKernel 架构

![AutoKernel 架构](architecture.png)

AutoKernel分为三个模块：
* 算子生成器: 

  该模块使用了开源项目[Halide](https://github.com/halide/Halide)；Halide是业界广泛使用的自动代码生成项目，它首次提出将计算和调度分离。该模块的输入是和硬件无关的算子计算描述，输出是相应后端的优化汇编代码/目标文件。

* 自动搜索模块：
  
  该模块可以通过最优化算法/搜索算法/机器学习/强化学习搜索出相应后端的最优算子的调度策略参数（该模块仍在开发中）；
* 算子部署插件（ AutoKernel Plugin）：
  
  [Tengine](https://github.com/OAID/Tengine)是OPEN AILAB开源的深度学习推理框架，实现了AI算法在不同硬件的快速高效部署。该模块实现了将自动生成的优化算子代码以plugin的形式一键集成到[Tengine](https://github.com/OAID/Tengine)中，实现自动优化算子的一键部署；
