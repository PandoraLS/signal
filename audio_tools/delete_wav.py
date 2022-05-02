# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 15:58
# @Author  : seenli
# @File    : delete_wav.py


import os, fnmatch
import random
import numpy as np

def process():
    wav_dir = r"C:\datasets\read_speech_clean16k"
    wav_list = fnmatch.filter(os.listdir(wav_dir), "*.wav")
    print('len of wav_list', len(wav_list))

    delete_list = np.random.choice(wav_list, 10000, replace=False) # 不放回抽样

    for wav_name in delete_list:
        wav_path = os.path.join(wav_dir, wav_name)
        os.remove(wav_path)



if __name__ == '__main__':
    print()
    process()