# -*- coding: utf-8 -*-
# @Time    : 2022/4/30 14:25
# @Author  : seenli


"""
Time Domain Adaptive Filters
Least Mean Squares Filter (LMS)
"""

import numpy as np
import librosa
import soundfile as sf

def lms(x, d, N = 4, mu = 0.1):
  """
  :param x: farend
  :param d: nearmic
  :param N: N阶滤波器
  :param mu: 梯度下降的步长
  :return:
  """
  nIters = min(len(x),len(d)) - N
  u = np.zeros(N) # used_x
  w = np.zeros(N)
  e = np.zeros(nIters)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n] # u[0, 1, 2, ...] = x[n, n-1, n-2, ...]
    e_n = d[n] - np.dot(u, w)
    w = w + mu * e_n * u
    e[n] = e_n
  return e

def main():
  x, sr = librosa.load('samples/x.wav', sr=8000) # farend
  d, sr = librosa.load('samples/d.wav', sr=8000) # nearmic
  e = lms(x, d, N=256, mu=0.1)
  e = np.clip(e, -1, 1)
  sf.write('samples/lms.wav', e, sr, subtype='PCM_16')

if __name__ == '__main__':
    main()