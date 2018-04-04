径向基神经网络python
data.txt：分类测试数据集

kmeans.py	    ------Kmean聚类代码，用来确定中心点
	参数：
	------k：为聚类中心数
	------max_iter：算法迭代次数（默认1000）基本不用设置，一般100内就收敛

RBFN.py     	------RBFN算法封装实现
	参数：
	------k：为KMean聚类中心数（默认10，K的个数对模型的性能有影响，可以进行调节）
	------delta：高斯函数中的扩展参数（默认0.1，可以进行调节）
	
RBFN_test.py	------对RBFN算法进行测试
	通过一个二分类数据集和经典多分类Iris数据集进行测试


对于模型的使用:
	1、导入模型
	from RBFN import RBFNet
	
	2、实例化模型
	rbf = RBFNet(k=10, delta=0.1) #可以通过交叉验证进行参数选择
	
	3、通过训练数据进行训练
	rbf.fit(x_train, y_train) #x_train, y_train都必须为array数据
	
	4、对测试集进行预测
	prediction = rbf.predict(x_test) #x_test必须为array数据

































