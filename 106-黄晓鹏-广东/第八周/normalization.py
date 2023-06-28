# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


#标准化

l=[-10, 5, 5, 6, 6,
   6, 7, 7, 7, 7,
   8, 8, 8, 8, 8,
   9, 9, 9, 9, 9,
   9, 10, 10, 10,10,
   10, 10, 10, 11, 11,
   11, 11, 11, 11, 12,
   12, 12, 12, 12, 13,
   13, 13, 13, 14, 14,
   14, 15, 15, 30]


def MyNormalization1(x):
    return [(float(i)-np.mean(x)) / (max(x) - min(x)) for i in x]

def MyNormalization2(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

n1 = MyNormalization1(l)
n2 = MyNormalization2(l)
print(n1)
print(n2)


cs=[]
for i in l:
    c=l.count(i)
    cs.append(c)

plt.plot(l,cs)
plt.plot(n1,cs)
plt.plot(n2,cs)
plt.show()