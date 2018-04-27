# -*- coding:utf-8 _*-
import numpy as np
from numpy import mat,zeros,nonzero,multiply
from sklearn import datasets
from sklearn import metrics
#from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def loadData(file_path):
    # 读入特定的圆形数据，用于核方法
    X = np.loadtxt(file_path, delimiter='\t', dtype=float)
    Y = X[:,-1]
    X = X[:,:-1]
    return X,Y

def selectJrand(i,m):
    # 在[0,m)中随机选择一个不等于i的数
    j = i
    while(i == j):
        j = np.random.randint(m)
    return j

def clipAlpha(alpha,high,low):
    # 使得alpha在[low,high]之内
    if alpha > high:
        alpha = high
    if alpha < low:
        alpha = low
    return alpha

def kernelTrans(X,row,kTup):
    # 这里X一般是m*n的矩阵或者m*1的列矩阵，row是1*n的行矩阵，返回m*1的矩阵
    m,n = X.shape
    K = mat(zeros((m,1)))
    if kTup[0] == 'linear':
        K = X*row.T
    elif kTup[0] == 'rbf':
        for i in range(m):
            deltaRow = X[i]-row
            K[i] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('The kernel %s is not recognized.' % kTup[1])
    return K


class optStruct(object):
    """用于存储计算中的一些变量的数据结构:optStruct"""
    def __init__(self, X_Mat, Y_Mat, C, toler, kTup):
        self.X = X_Mat
        self.Y = Y_Mat
        self.C = C
        self.toler = toler
        self.m = X_Mat.shape[0] # 数据的个数
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        # 创建一个m大小的元组缓存误差E值，第一维表示是否有效，第二个表示E的值
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i], kTup)

def calcEk(oS, k):
    # 计算第k个实例对应的Ek值
    fxk = float(multiply(oS.alphas, oS.Y).T*oS.K[:,k])+oS.b
    Ek = fxk-float(oS.Y[k])
    return Ek

def selectJ(i,oS,Ei):
    # 根据第i个alpha选取第j个alpha
    maxJ = -1;maxDeltaE = 0;Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0] # 得到oS.eCache中有效的E值对应的下标元组
    if len(validEcacheList) > 1:
        for j in validEcacheList:
            if j == i:
                continue
        tmpEj = calcEk(oS,j)
        deltaE = abs(Ei - tmpEj)
        if deltaE > maxDeltaE:
            maxDeltaE = deltaE; maxJ = j; Ej = tmpEj
        return maxJ,Ej      
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j,Ej

def updateEk(oS,k):
    # 更新oS中缓存的Ek值
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i,oS):
    Ei = calcEk(oS, i)
    if ((oS.Y[i]*Ei < -oS.toler) and (oS.alphas[i] < oS.C)) or ((oS.Y[i]*Ei > oS.toler) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy() # 因为oS.alphas是一个矩阵，传进来的是引用，
        alphaJold = oS.alphas[j].copy() # 所以这里要用拷贝，直接等只是一个引用
        if oS.Y[i] != oS.Y[j]:
            low = max(0, oS.alphas[j]-oS.alphas[i])
            high = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])    
        else:
            low = max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
            high = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if low == high:
            # print "low == high"
            return 0
        # 计算eta值
        eta = 2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if eta >= 0:
            # print "eta >= 0"
            return 0
        # 求解新的alphas[j]
        oS.alphas[j] -= oS.Y[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], high, low)
        updateEk(oS, j)
        # 求解新的alphas[i]
        if abs(oS.alphas[j]-alphaJold) < 0.00001:
            # print 'j is not moving enough'
            return 0
        oS.alphas[i] += oS.Y[j]*oS.Y[i]*(alphaJold-oS.alphas[j])
        updateEk(oS, i)
        # 计算对应的b1,b2,用于更新b
        b1 = oS.b - Ei - oS.Y[i]*oS.K[i,i]*(oS.alphas[i]-alphaIold) - oS.Y[j]*oS.K[i,j]*(oS.alphas[j]-alphaJold)
        b2 = oS.b - Ej - oS.Y[i]*oS.K[i,i]*(oS.alphas[i]-alphaIold) - oS.Y[j]*oS.K[i,j]*(oS.alphas[j]-alphaJold)
        if (0 < oS.alphas[i]) and (oS.alphas[i] < oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0

def smo(X,Y,C,toler,maxIter,kTup=('linear',0)):
    oS = optStruct(mat(X), mat(Y).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True # True表示需要遍历整个数据集
    alphaPairsChanged = 0 # 表示发生改变的alpha对个数
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                # print 'fullset, iter:%d,i:%d,pairs changed:%d' % (iter,i,alphaPairsChanged)
        else:
            # 得到所有在(0,C)范围的alphas对应的下标
            nonBoundIdxs = nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0] 
            for i in nonBoundIdxs:
                alphaPairsChanged += innerL(i, oS)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                # print 'nonBound, iter:%d,i:%d,pairs changed:%d' % (iter,i,alphaPairsChanged)
        iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print ('iteration number: %d' % iter)
    return oS.alphas,oS.b

def calcWs(X,Y,alphas):
    X = mat(X)
    Y = mat(Y).transpose()
    w = X.T*multiply(alphas, Y)
    return w

def predict(X,Y,w,b):    
    tmp = X*w+b
    preY = [1 if float(t) > 0 else -1 for t in tmp]
    return preY

def plotPoints(X,Y):
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []
    for i,y in enumerate(Y):
        if y == -1:
            neg_x.append(X[i][0])
            neg_y.append(X[i][1])
        else:
            pos_x.append(X[i][0])
            pos_y.append(X[i][1])
    plt.scatter(neg_x, neg_y, c='r', marker='o')
    plt.scatter(pos_x, pos_y, c='g', marker='o')

def plotDecesion(X,Y,decision_func):
    plotPoints(X, Y)
    x_min = np.min(X[:,0].ravel())
    x_max = np.max(X[:,0].ravel())
    x = np.arange(x_min,x_max,0.01)
    y = decision_func(x)
    plt.plot(x,y)
    plt.show()

def testLinearSvm(X,Y):
    alphas,b = smo(X, Y, 1.0, 0.001, 40)
    w = calcWs(X, Y, alphas)
    preY = predict(X, Y, w, b)
    print (metrics.classification_report(Y, preY))
    w = w.A.ravel() # 因为w,b都是matrix类型,所以要先转换
    b = float(b)
    f = lambda x : -float(w[0])*x/float(w[1])-b/float(w[1])
    plotDecesion(X, Y, f)
    plt.figure()

def testSVM(X,Y,kTup):
    alphas,b = smo(X, Y, 1.0, 0.001, 40, kTup=kTup)
    svIdx = nonzero(alphas.A>0)[0] # 得到支持向量对应的下标
    X = mat(X)
    Y = mat(Y).transpose() # 将X,Y转化成矩阵
    m,n = X.shape
    svs = X[svIdx] # 得到支持向量
    svY = Y[svIdx] # 得到支持向量对应的分类
    print ('there are %d support_vectors' % svs.shape[0])
    # 进行分类 
    preY = []
    for i in range(m):
        kernelEval = kernelTrans(svs, X[i], kTup)
        predict = kernelEval.T * multiply(svY, alphas[svIdx]) + b
        y = -1 if predict < 0 else 1
        preY.append(y)
    print (metrics.classification_report(Y.A.ravel(), preY))
    X = X.A
    Y = Y.A.ravel() # 把X,Y从矩阵变回普通的数组
    # 画出所有的点
    plotPoints(X, Y)
    # 标出支持向量
    plt.scatter(svs[:, 0], svs[:, 1],s=80, facecolors='none')
    # 画出决策边界
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    plt.show()
    plt.figure()


if __name__ == "__main__":
    X,Y = loadData('testSet.txt')
    # X,Y = loadData()
    # testLinearSvm(X, Y)
    # kernel = ('linear',10.0)
    kTup = ('rbf', 0.5) # 对于核函数来说，sklearn的auto对应1/n_feature，试了一下，效果很好
    testSVM(X, Y, kTup)
    
    # 用sklearn的SVC线性分类器
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf')
    clf.fit(X,Y)
    preY = clf.predict(X)
    print (metrics.classification_report(Y, preY))
    # print 'sklearn: '
    # print clf.coef_,clf.intercept_
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    # 画出对应的点
    plotPoints(X, Y)
    # 画出对应的支持向量
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],s=80, facecolors='none')
    plt.show()
    
    
    
    
    
    
    
    
