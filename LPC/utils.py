# -*- coding: utf-8 -*-
# @Time    : 2021/11/14 15:24
# @Author  : seenli
# @File    : utils.py

"""
https://github.com/kunigami/kunigami.github.io/blob/master/blog/code/2021-05-13-lpc-in-python/lpc.ipynb
"""
import scipy.io.wavfile
import numpy as np
from math import floor
import scipy.signal as signal
from scipy.signal import lfilter, resample
from scipy.signal.windows import hann
from numpy.random import randn
import matplotlib.pyplot as plt

sample_rate = 8000
sym = False # periodic
window_size = floor(0.03*sample_rate)

window = hann(window_size, sym)

t = np.array(range(window_size))

fig = plt.figure(figsize=(18, 5))
plt.plot(t, window, 'b')

plt.show()