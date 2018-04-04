# -*- coding: utf-8 -*-
"""
用Python进行，回归分析，多重共线性检验，异方差处理，自相关处理。
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import matplotlib.mlab as mlab

#导入数据
data = pd.read_excel('数据.xlsx', sheetname='整合数据', header=None)
data = data.T

#将数据进行转换备份
data.to_csv('data.csv', header=None, index=False)
data = pd.read_csv('data.csv', encoding='GBK')
data = data[::-1] #年份从小到大

#简化变量名称（一一对应）
'''指标, 公路客运量, 公路里程, 等级公路里程, 高速等级路公路里程, 一级等级公路里程, 二级等级公路里程, 城镇人口, 
国民总收入, 国民总收入指数, 铁路客运量(万人), 民用航空客运量(万人), 公路营运汽车拥有量(万辆), 私人汽车拥有量(万辆)'''
'''year ,Road passenger traffic , mileage1, mileage2, mileage3, mileage4, mileage5, population, income, 
Income rate, Railway passenger volume, Civil aviation passenger traffic, The ownership of highway operating vehicles,
Private car ownership'''
'''YE ,RPT, m1, m2, m3, m4, m5, pL, IC, IR, RPV, CAPT, OOHOV, PCO'''

data.columns = ['YE' ,'RPT', 'm1', 'm2', 'm3', 'm4', 'm5', 'PL', 'IC', 'IR', 'RPV', 'CAPT', 'OOHOV', 'PCO']
print(data.head())

#***************************************************************************************
#*************************************多重共线性检验**************************************
#***************************************************************************************
print('***************************多重共线性检验*********************************')
'''
多重共线性是指线性回归模型中的解释变量之间由于存在精确相关关系或高度相关关系而使模型估计失真或难以估计准确。
当自变鱼之间高度相关时, 回归系数表现出不确定性, 从而使回归系数的标准差大大增加(模型的预测不稳定性增加)。
从应用角度看, 由于多共线性的存在, 如果仅从回归系数的经济意义出发去解释经济现象。
'''
'''
判断方法：（理论解释参考资料）
方法1：方差膨胀因子(VIF)，如果VIF大于10，则说明变量存在多重共线性。
方法2：相关系数矩阵法，通过观察相关系数矩阵中变量的相关性进行判断。
解决方法：逐步回归法
'''

# ====== 1、方差膨胀因子(VIF)完成多重共线性检验的判断 ======
# 将因变量公路客运量，自变量和截距项（值为1的1维数组）以数据框的形式组合起来
c = list(data.columns[1:])
y, X = dmatrices(c[0]+'~'+c[1]+'+'+c[2]+'+'+c[3]+'+'+c[4]+'+'+c[5]+'+'+c[6]
                    +'+'+c[7]+'+'+c[8]+'+'+c[9]+'+'+c[10]+'+'+c[11]+'+'+c[12], 
                    data=data.drop('YE', axis=1), return_type='dataframe')
# 构造空的数据框
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif.round(1))

# ====== 2、相关系数矩阵法完成多重共线性检验的判断 ======
#计算相关系数矩阵
correlation = data.drop('YE', axis=1).corr()
print(correlation) 

#相关矩阵图
plt.figure()
fig = plt.subplots(figsize=(10, 10))
sns.heatmap(correlation, vmax=0.8, square=True, annot=True)
plt.savefig('多重共线性检验-各变量之间的相关系数矩阵热力图.png')
plt.show()

'''
结论：从相关系数矩阵，和VIF值都可以看出变量间存在共线性问题
    VIF Factor   features
0      38320.0  Intercept
1         65.6         m1
2        517.3         m2
3       2897.9         m3
4       4407.2         m4
5       3356.4         m5
6       4180.9         pL
7       1137.9         IC
8          3.5         IR
9        261.1        RPV
10       960.1       CAPT
11        34.3      OOHOV
12      1897.0        PCO
分析：从方差膨胀因子(VIF)，和各变量之间的相关系数矩阵热力图都可以看出变量之间存在共线性问题。
'''

# ====== 逐步回归的办法，去解决多重共线性问题 ======
#第一步分别作RPT对其他变量的一元回归，对可决定系数R-squared进行排序

#定义多元回归函数
def MulitiLinear_regressionModel(df, y, x):
    model = sm.formula.ols(y + '~' + x, data=df).fit()
    print((model.summary()))
    return model._results.params

#分别作RPT对其他变量的一元回归
for col in data.columns[2:]:
    print('%%%%%%%%%%%%%%%%%%%%%%%'+ 'RPT' + '~' + col +'%%%%%%%%%%%%%%%%%%%%%%%')
    MulitiLinear_regressionModel(data, 'RPT', col)

'''
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.372  顺序：1
%%%%%%%%%%%%%%%%%%%%%%%RPT~m2%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.321  顺序：3
%%%%%%%%%%%%%%%%%%%%%%%RPT~m3%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.203  顺序：7
%%%%%%%%%%%%%%%%%%%%%%%RPT~m4%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.234  顺序：5
%%%%%%%%%%%%%%%%%%%%%%%RPT~m5%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.334  顺序：2
%%%%%%%%%%%%%%%%%%%%%%%RPT~PL%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.285  顺序：4
%%%%%%%%%%%%%%%%%%%%%%%RPT~IC%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.185  顺序：8
%%%%%%%%%%%%%%%%%%%%%%%RPT~IR%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.025  顺序：12
%%%%%%%%%%%%%%%%%%%%%%%RPT~RPV%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.104  顺序：10
%%%%%%%%%%%%%%%%%%%%%%%RPT~CAPT%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.163  顺序：9
%%%%%%%%%%%%%%%%%%%%%%%RPT~OOHOV%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.216  顺序：6
%%%%%%%%%%%%%%%%%%%%%%%RPT~PCO%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:         0.079  顺序：11
以可决系数的大小为依据，对解释变量的重要程度进行排序，依次是：
m1, m5, m2, PL, m4, OOHOV, m3, IC, CAPT, RPV, PCO, IR
'''

#第二步：以"RPT ~ m1"模型为基础，根据可决定系数的大小加入变量
c = ['m1','m5','m2','PL','m4','OOHOV','m3','IC','CAPT','RPV','PCO','IR']
count = 0
for i in range(len(c)): 
    count += 1
    colx = '+'.join(c[0:count])
    print('%%%%%%%%%%%%%%%%%%%%%%%'+ 'RPT' + '~' + colx +'%%%%%%%%%%%%%%%%%%%%%%%')
    MulitiLinear_regressionModel(data, 'RPT', colx)
    
'''
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.372
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.375
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5+m2%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.376
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5+m2+PL%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.657
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5+m2+PL+m4%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.776
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5+m2+PL+m4+OOHOV%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.800
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5+m2+PL+m4+OOHOV+m3%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.801
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5+m2+PL+m4+OOHOV+m3+IC%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.835
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5+m2+PL+m4+OOHOV+m3+IC+CAPT%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.835
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5+m2+PL+m4+OOHOV+m3+IC+CAPT+RPV%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.838
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5+m2+PL+m4+OOHOV+m3+IC+CAPT+RPV+PCO%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.847
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+m5+m2+PL+m4+OOHOV+m3+IC+CAPT+RPV+PCO+IR%%%%%%%%%%%%%%%%%%%%%%%
Dep. Variable:                    RPT   R-squared:                       0.847

结论：逐步回归的办法模型的R-squared变化过程为：
['m1','m5','m2','PL','m4','OOHOV','m3','IC','CAPT','RPV','PCO','IR']
0.372, 0.375, 0.376, 0.657, 0.776, 0.800, 0.801, 0.835, 0.835, 0.838, 0.847, 0.847,

去除逐步回归法中模型影响很小的变量（['m5','m2','m3','CAPT','IR']）
同时可以看到这也符合一些常识：'m5','m2','m3'和'm1','m4'也是变量意义上相近的（看原来变量）
0.372, 0.657, 0.776, 0.800, 0.835, 0.847
['m1','PL','m4','OOHOV','IC','RPV','PCO']
'''

colx = '+'.join(['m1','PL','m4','OOHOV','IC','RPV','PCO'])
coly = data.columns[1]
print('%%%%%%%%%%%%%%%%%%%%%%%'+ coly + '~' + colx + '%%%%%%%%%%%%%%%%%%%%%%%')
fit = sm.formula.ols(coly + '~' + colx, data=data).fit()
print(fit.summary())

'''
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+PL+m4+OOHOV+IC+RPV+PCO%%%%%%%%%%%%%%%%%%%%%%%
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    RPT   R-squared:                       0.817
Model:                            OLS   Adj. R-squared:                  0.711
Method:                 Least Squares   F-statistic:                     7.663
Date:                Sun, 01 Apr 2018   Prob (F-statistic):            0.00122
Time:                        21:36:53   Log-Likelihood:                -280.71
No. Observations:                  20   AIC:                             577.4
Df Residuals:                      12   BIC:                             585.4
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept  -3.264e+06   6.43e+06     -0.508      0.621     -1.73e+07  1.07e+07
m1         -4948.5716   5031.001     -0.984      0.345     -1.59e+04  6013.039
PL            87.8823    171.374      0.513      0.617      -285.509   461.273
m4         -1.211e+05   9.62e+05     -0.126      0.902     -2.22e+06  1.97e+06
OOHOV       -838.0359   1023.257     -0.819      0.429     -3067.522  1391.450
IC            17.5820      5.025      3.499      0.004         6.633    28.531
RPV            7.6847     21.464      0.358      0.727       -39.082    54.451
PCO         -790.6724    261.681     -3.022      0.011     -1360.827  -220.518
==============================================================================
Omnibus:                        1.032   Durbin-Watson:                   1.704
Prob(Omnibus):                  0.597   Jarque-Bera (JB):                0.145
Skew:                           0.120   Prob(JB):                        0.930
Kurtosis:                       3.340   Cond. No.                     3.12e+07
==============================================================================
建立的模型中：R-squared = 0.817 模型可靠
'''

#***************************************************************************************
#*************************************异方差处理**************************************
#***************************************************************************************
'''
在线性回归建模中，如果模型表现的非常好的话，那么残差与拟合值之间不应该存在某些明显的关系或趋势。
如果模型的残差确实存在一定的异方差的话，会导致估计出来的偏回归系数不具备有效性，甚至导致模型的
预测也不准确。所以，建模后需要验证残差方差是否具有齐性，检验的方法有两种，一种是图示法，一种是
统计验证法。（理论解释参考资料）
'''
print('*****************异方差处理****************')

# ====== 1、图示法完成方差齐性的判断 ======
# 标准化残差与预测值之间的散点图
plt.figure()
plt.scatter(fit.predict(), (fit.resid-fit.resid.mean())/fit.resid.std())
plt.xlabel('预测值')
plt.ylabel('标准化残差')

# 添加水平参考线
plt.axhline(y = 0, color = 'r', linewidth = 2)
plt.savefig('异方差处理-标准化残差与预测值之间的散点图.png')
plt.show()

'''
从图中看，并没发现明显的规律或趋势（判断标准：如果残差在参考线两侧均匀分布，
则意味着异方差性较弱；而如果呈现出明显的不均匀分布，则意味着存在明显的异方差性。）
，故可以认为没有显著的异方差性特征。
'''

# ====== 2、统计法完成方差齐性的判断 ======
# White's Test
print('White\'s Test')
print(sm.stats.diagnostic.het_white(fit.resid, exog = fit.model.exog))
'''
het_white函数返回值说明：
Returns
-------
lm : float
    lagrange multiplier statistic
lm_pvalue :float
    p-value of lagrange multiplier test
fvalue : float
    f-statistic of the hypothesis that the error variance does not depend
    on x. This is an alternative test variant not the original LM test.
f_pvalue : float
    p-value for the f-statistic
'''
# Breusch-Pagan
print('Breusch-Pagan')
print(sm.stats.diagnostic.het_breushpagan(fit.resid, exog_het = fit.model.exog))
'''
het_breushpagan函数返回值说明：
Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue :float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x
    f_pvalue : float
        p-value for the f-statistic
'''

'''
结果：
White's Test :P-value = 0.39457818208600143
(20.0, 0.39457818208600143, 0.0, nan)
Breusch-Pagan: P-value = 0.40071494593778484
(12.574442783165573, 0.40071494593778484, 0.98781699599377149, 0.53091715919089377)

结论：不论是White检验还是Breush-Pagan检验，P值都远远大于0.05这个判别界限，
即接受原假设（残差方差为常数的原假设），认为残差满足齐性这个假设（即不存在异方差情况）。
（理论解释参考资料）
'''

#***************************************************************************************
#****************************************自相关处理***************************************
#***************************************************************************************
'''
（理论解释参考资料）
（1）Durbin-Watson 检验法：当D.W.值在2左右时，模型不存在一阶自相关。
（2）图示法：残差图
'''
print('#*************************自相关处理********************************')
# ======1、Durbin-Watson 检验法（用上面得到的模型进行训练）======
colx = '+'.join(['m1','PL','m4','OOHOV','IC','RPV','PCO'])
coly = data.columns[1]
print('%%%%%%%%%%%%%%%%%%%%%%%'+ coly + '~' + colx + '%%%%%%%%%%%%%%%%%%%%%%%')
fit = sm.formula.ols(coly + '~' + colx, data=data).fit()
print(fit.summary())

'''
%%%%%%%%%%%%%%%%%%%%%%%RPT~m1+PL+m4+OOHOV+IC+RPV+PCO%%%%%%%%%%%%%%%%%%%%%%%
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    RPT   R-squared:                       0.817
Model:                            OLS   Adj. R-squared:                  0.711
Method:                 Least Squares   F-statistic:                     7.663
Date:                Sun, 01 Apr 2018   Prob (F-statistic):            0.00122
Time:                        21:36:53   Log-Likelihood:                -280.71
No. Observations:                  20   AIC:                             577.4
Df Residuals:                      12   BIC:                             585.4
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept  -3.264e+06   6.43e+06     -0.508      0.621     -1.73e+07  1.07e+07
m1         -4948.5716   5031.001     -0.984      0.345     -1.59e+04  6013.039
PL            87.8823    171.374      0.513      0.617      -285.509   461.273
m4         -1.211e+05   9.62e+05     -0.126      0.902     -2.22e+06  1.97e+06
OOHOV       -838.0359   1023.257     -0.819      0.429     -3067.522  1391.450
IC            17.5820      5.025      3.499      0.004         6.633    28.531
RPV            7.6847     21.464      0.358      0.727       -39.082    54.451
PCO         -790.6724    261.681     -3.022      0.011     -1360.827  -220.518
==============================================================================
Omnibus:                        1.032   Durbin-Watson:                   1.704
Prob(Omnibus):                  0.597   Jarque-Bera (JB):                0.145
Skew:                           0.120   Prob(JB):                        0.930
Kurtosis:                       3.340   Cond. No.                     3.12e+07
==============================================================================

对于第一种方法：Durbin-Watson = 1.704，可以看出不存在自相关（理论解释参考资料）
'''

# ======2、图示法：残差图======
plt.figure()
plt.plot(fit.resid)
plt.ylabel('残差')
plt.savefig('异方差处理-残差图.png')
plt.show()

'''
结论：根据残差图可以看出来不存在自相关（理论解释参考资料）
'''

#***************************************************************************************
#****************************************最终模型建立*************************************
#***************************************************************************************
colx = '+'.join(['m1','PL','m4','OOHOV','IC','RPV','PCO'])
coly = data.columns[1]
print('%%%%%%%%%%%%%%%%%%%%%%%'+ coly + '~' + colx + '%%%%%%%%%%%%%%%%%%%%%%%')
fit = sm.formula.ols(coly + '~' + colx,data=data).fit()
print(fit.summary())

'''
通过：多重共线性检验   异方差处理   自相关处理 这几个部分的处理可以得到回归模型参数为：
                 coef    
---------------------
Intercept  -3.264e+06  
m1         -4948.5716  
PL            87.8823 
m4         -1.211e+05  
OOHOV       -838.0359 
IC            17.5820   
RPV            7.6847  
PCO         -790.6724  
即模型为：
RPT = -3.264e+06 - 4948.5716*m1 + 87.8823*PL - 1.211e+05*m4 - 838.0359*OOHOV + 17.5820*IC + 7.6847*RPV - 790.6724*PCO 
'''








