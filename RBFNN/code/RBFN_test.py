# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from RBFN import RBFNet


#数据集导入
def getdata(): #分类测试数据集
    data = pd.read_table('data.txt', sep=' ', header=None)
    X = data.drop(2, axis=1).values #从pandas得到array数据（模型需要数据格式）
    y = data[2].values #从pandas得到array数据（模型需要数据格式）
            
    return X, y
   
def getIris(): #经典Iris分类数据集
    np.random.seed(2018)
    data = datasets.load_iris()
    X = data.data
    y = data.target

    return X, y


################################测试 RBFN#####################################
print('###########################################################')
#在第一个数据集上测试
X, y = getdata()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rbf = RBFNet(k=10, delta=0.1)
rbf.fit(x_train, y_train)

print('result', y_test)
prediction = rbf.predict(x_test)

#由于RBFN输出结果为数值型（拟合结果），所以对于分类需要对结果进行规约（数据label为：-1,1）
prediction = np.round(prediction)
print('prediction', prediction) 

print('==========================================================')
print('测试 RBFN getdata accuracy_score', accuracy_score(y_test, prediction))
print('==========================================================')

print('###########################################################')
#在第二个数据集上测试
X, y = getIris()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rbf = RBFNet(k=10, delta=0.1)
rbf.fit(x_train, y_train)

print('result', y_test)
prediction = rbf.predict(x_test)
print('prediction', prediction)

#由于RBFN输出结果为数值型（拟合结果），所以对于分类需要对结果进行规约（数据label为：0,1,2）
for i in range(len(prediction)):
    if prediction[i] <= 0.5:
        prediction[i] = 0
    if (prediction[i] > 0.5) & (prediction[i] <= 1.5):
        prediction[i] = 1
    if (prediction[i] > 1.5) & (prediction[i] <= 2.5):
        prediction[i] = 2  

print('prediction', prediction)

print('==========================================================')
print('测试 RBFN getIris accuracy_score', accuracy_score(y_test, prediction))
#print(u'测试 RBFN getIris accuracy_score', accuracy_score(y_test, prediction))
print('==========================================================')



'''
==========================================================
测试 RBFN getdata accuracy_score 0.95
==========================================================

==========================================================
测试 RBFN getIris accuracy_score 0.9777777777777777
==========================================================
'''







