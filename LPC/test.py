# -*- coding: utf-8 -*-
# @Time    : 2021/11/14 16:04
# @Author  : seenli
# @File    : test.py

import numpy as np
import warnings
warnings.filterwarnings("ignore")

x = np.arange(10)
print('x', x)
b = x[1:].T
print('b.T', b)

xz = np.concatenate((x[::-1], np.zeros(6)))
print('xz', xz)

n = len(x)
print('n:', n)
X = np.zeros((n-1, 6))
for i in range(n-1):
    offset = n-1-i
    X[i, :]=xz[offset: offset+6]

print('X', X)

# y = mx + c,这里m对应a，c舍去
a  = np.linalg.lstsq(X, b)[0]
print('a',a)

e = b.T - np.dot(X, a) # 计算误差
print('e', e)

g = np.var(e)
print('g', g)
