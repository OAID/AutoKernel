todo:
- decouple cost_model from current codes
- prepare for merging state-of-the-art cost models 



# 6月21日更新
## 变更：
1. 添加了读取feature和runtime的功能：load_sample.py
2. 跑通了前向传播和反向传播：cost_model.py
3. 主函数：retrain_cost_model.py
## 结果：
正向传播预测的结果为3.47，真实值为1.61.
## 使用方法：
```
python3 retrain_cost_model.py
```

