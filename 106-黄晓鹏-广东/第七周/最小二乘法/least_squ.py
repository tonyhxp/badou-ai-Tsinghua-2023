# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import leastsq

'''
最小二乘法

'''



data = pd.read_csv('train_data.csv',sep='\s*,\s*',engine='python')
X=data['X'].values
Y=data['Y'].values

plt.figure()
plt.scatter(X,Y,color='blue',label='dataaa',linewidths=2)
plt.show()

# model
def getModel(m,x):
    k,b= m
    return k*x + b

#err
def getErr(m,x,y):
    return getModel(m, x) - y

#init
p = np.random.randn(2)

res = leastsq(getErr,p,args=(X,Y))

k,b = res[0]
print(k)
print(b)







