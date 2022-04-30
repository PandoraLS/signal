# -*- coding: utf-8 -*-
# @Time    : 2022/4/30 23:58
# @Author  : seenli


import numpy as np
import librosa
import soundfile as sf
import pyroomacoustics as pra

x, sr = librosa.load('samples/female.wav', sr=8000) # 远端参考
d, sr = librosa.load('samples/male.wav', sr=8000) # 近端语音

rt60_tgt = 0.08
room_dim = [2, 2, 2]

e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)
room.add_source([1.5, 1.5, 1.5])
room.add_microphone([0.1, 0.5, 0.1])
room.compute_rir()
rir = room.rir[0][0]
rir = rir[np.argmax(rir):]

y = np.convolve(x, rir) # 远端参考卷为回声(什么是卷积?)
scale = np.sqrt(np.mean(x ** 2)) / np.sqrt(np.mean(y ** 2))
y = y * scale

L = max(len(y), len(d))
y = np.pad(y, [0, L - len(y)]) # 在y尾部补 L-len(y) 个零
d = np.pad(d, [L - len(d), 0]) # 在d头部补 L-len(d) 个零
x = np.pad(x, [0, L - len(x)])
d = d + y # d-> nearmic

sf.write('samples/x.wav', x, sr, subtype='PCM_16') # farend
sf.write('samples/d.wav', d, sr, subtype='PCM_16') # nearmic
sf.write('samples/y.wav', y, sr, subtype='PCM_16') # echo
sf.write('samples/rir.wav', rir, sr, subtype='PCM_16') # rir

