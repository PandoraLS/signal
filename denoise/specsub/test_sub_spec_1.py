import librosa
from librosa.core.spectrum import amplitude_to_db
import numpy as np
import soundfile as sf


if __name__ == "__main__":
    clean_wav_file = "sf1_cln.wav"
    clean,fs = librosa.load(clean_wav_file,sr=None)  # 这里sr=None如果没写的话，librosa会自动重采样为22k
    print(fs)

    noisy_wav_file = "sf1_n0L.wav"
    noisy,fs = librosa.load(noisy_wav_file,sr=None)

    # 计算 nosiy 信号的频谱
    S_noisy = librosa.stft(noisy,n_fft=256, hop_length=128, win_length=256)  # D x T  D=129,T=256(帧长)
    D,T = np.shape(S_noisy)
    Mag_noisy= np.abs(S_noisy)
    Phase_nosiy= np.angle(S_noisy)
    Power_nosiy = Mag_noisy**2
    print(fs)
    # 估计噪声信号的能量
    # 由于噪声信号未知 这里假设 含噪（noisy）信号的前30帧为噪声
    Mag_nosie = np.mean(np.abs(S_noisy[:,:30]),axis=1,keepdims=True)
    Power_nosie = Mag_nosie**2
    Power_nosie = np.tile(Power_nosie,[1,T])

    # 能量减
    Power_enhenc = Power_nosiy-Power_nosie
    Power_enhenc[Power_enhenc<0]=0 # 保证能量大于0，但可能会出现比较陡峭的突然衰减
    Mag_enhenc = np.sqrt(Power_enhenc)

    # 幅度减
    # Mag_enhenc = np.sqrt(Power_nosiy) - np.sqrt(Power_nosie)
    # Mag_enhenc[Mag_enhenc<0]=0
    # 无论是用幅度谱减还是用能量谱减，效果其实差不多

    # 对信号进行恢复
    S_enhec = Mag_enhenc*np.exp(1j*Phase_nosiy)
    enhenc = librosa.istft(S_enhec, hop_length=128, win_length=256)
    sf.write("enhce_n0L_1.wav",enhenc,fs)
    print(fs)
    # 绘制谱图
