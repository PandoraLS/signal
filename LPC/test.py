# -*- coding: utf-8 -*-
# @Time    : 2021/11/14 16:04
# @Author  : seenli
# @File    : test.py

import numpy as np
from scipy.signal import lfilter, resample
import warnings
warnings.filterwarnings("ignore")

# 信号x
x = np.arange(20)
print('x', x)
b = x[1:].T
print('b.T', b)

xz = np.concatenate((x[::-1], np.zeros(6)))
print('xz', xz)

n = len(x)
print('n:', n)
X = np.zeros((n-1, 6)) # 这里6对应p阶数，p极点数
for i in range(n-1):
    offset = n-1-i
    X[i, :]=xz[offset: offset+6]

print('X', X)

# LPC编码 获取a和g
a  = np.linalg.lstsq(X, b)[0]
print('a',a)

e = b.T - np.dot(X, a) # 计算误差
print('e', e)

g = np.var(e)
print('g', g)

# LPC解码，恢复x
src = np.sqrt(g) * np.random.randn(20, 1) # 这里20 对应信号长度 20，产生一个能量与g对等的随机信号
b2 = np.concatenate([np.array([-1]), a])
x_hat = lfilter([1], b2.T, src.T).T
print('x_hat', x_hat)