# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import accuracy_score
from pprint import pprint

#导入数据
data = pd.read_table('Font_dataset.txt', header=None, sep=',')

#特征数据和标签
X = data.drop(4, axis=1)
y = data[4]

from tree import DecisionTree
clf = DecisionTree()


print(u"*****在自己的决策树上进行10折交叉验证*****")
test_accuracy = []
L = X.shape[0]
kf = KFold(L, n_folds=10, random_state=2018)
count = 0
for train_index, test_index in kf:
    count += 1
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y.values[train_index], y.values[test_index]
    #训练
    clf.fit(X.values, y.values)
    #测试
    test_pre = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pre)
    test_accuracy.append(test_acc)
    print('%d test accuracy_score :%.4f' % (count,test_acc))
    
print('mean test accuracy_score :%.4f' % np.mean(test_accuracy))













