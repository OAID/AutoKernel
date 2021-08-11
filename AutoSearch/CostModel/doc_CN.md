# CostModel

CostModel是AutoSearch模块的重要组成部分，它通过一个小型神经网络来预测Halide算子策略的运行时间，并且能够自动生成训练数据对该神经网络进行训练。



## 快速使用

```python
cd tookit
from CostModel import autosampler

gen_path = '../generator/random_pipeline_generator.cpp'
demo_name = 'random_pipeline'
batch_num = 2
autosampler.retrain_cost_model(gen_path,demo_name,batch_num)

```





## 结构 

### costmodel.py

该文件是CostModel模块的核心部分，它包括了将数据导入，定义神经网络模型，前向传播计算Loss，反向传播优化模型权重的功能。这部分内容参考了adams2019中的cost_model_generator.cpp，将其改写成pytorch框架下的神经网络模型。

该脚本包括了：CSVDataset类和Net类以及用于训练模型的train_cost_model函数。

* CSVDataset类：

  继承torch中的Dataset类，在初始化实例时根据数据的相对路径导入数据。在训练函数train_cost_model中用于DataLoader的实例初始化。

  该类的属性有：

  * Xp：pipeline的特征数据。每个sample有39个。
  * Xs：schedule的特征数据。每个sample有40*7个。
  * y：  schedule的实测运行时间。

* Net类：

  继承nn.Module类，在初始化时定义了三个FC层。描述了整个神经网络的前向计算过程，最终计算出loss。

  该类的方法有：

  * network()：输入特征数据，进行神经网络部分计算。对应cost_model_generator.cpp源码当中的184-230行部分。
  * forword()：进一步根据输入的机器性能，如num_cores计算sample的cost。对应源码当中的232-384行部分。
  * loss_func()：计算所预测的cost与真实值y之间的loss。对应源码中的403-436行部分。

* train_cost_model()函数：

  1. CSVDataset类实例初始化，根据路径导入训练数据。

  2. Weights类实例初始化，根据路径向模型中导入用于训练的权重。
  3. 定义优化器后训练。
  4. 由于模型中的权重格式与Halide有区别，最后需要对权重进行后处理后导出到指定的路径和格式下。



### Weights.py

该文件定义了Weights类，用于权重的从.weight格式和json格式到字典和tensor之间的导入导出操作，以及载入模型对格式的预处理。

Weights类的方法有：

* load_from_model()：从Net类的实例，也就是神经网络模型中导入权重到Weights实例中。
* load_in_model()：将权重导入到神经网络模型中。
* load_from_file()：从.weight文件中读取权重值。
* save_to_json()：将权重导出为json文件。
* load_from_json()：读取json文件为权重。
* save_to_file()：将权重保存为.weight文件。



### sample_load.py

该文件将路径下的.sample文件读取出pipeline_feature，schedule_feature和runtime，并将其转化成torch中的tensor张量以用于训练。

该文件有以下两个函数：

* load_a_sample()：根据文件名读取一个sample文件。
* load_from_dir()：读取某一路径下所有的sample文件。



### autosampler.py

该文件能够对给定的算法描述文件调用Halide程序，生成其generator文件以及schedule文件，并进行benchmark，从而得到sample文件。

该文件中的retrain_cost_model()函数是CostModel模块中最终整合了整个训练流程的函数，它接收以下参数：

* gen_path：Halide语言描述的算子或算法文件的路径。
* demo_name：用户为算子或算法的命名，如matmul。
* batch_num：本次调用一共训练的batch数量。
* batch_size：一个batch下的sample数量，默认为16个。
* samples_dir：储存sample的文件夹，如果不存在则会新建，已存在则会在原有的batch序号后继续添加新的batch。默认为“./default_samples"
* weight_path：要训练的.weights文件的路径，若文件不存在则随机初始化，再进行训练。默认为“./random_init.weights"。
* learning_rate：优化器的学习率，默认为0.001。
* train_iters：对于每个batch，训练时迭代的次数，默认为200次。



