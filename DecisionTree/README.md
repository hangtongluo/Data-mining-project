
大神代码链接：https://github.com/wepe/MachineLearning/tree/master/DecisionTree

存在问题：如果测试集中某个样本的某个特征的值在训练集中没出现，则会造成训练出来的树的某个分支，对该样本不能分类，出现KeyError  

经过测试会出现用训练集训练测试集预测报错，用全数据集训练用部分数据集预测好着呢
