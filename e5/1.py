import numpy as np
import matplotlib.pyplot as plt

# 温度采样点
t = np.array([12.8, 12.9, 12.8, 12.8, 12.7, 11.1, 15.7, 17.4, 14.6, 12.5, 12.8, 12.5])

# 计算离散傅里叶变换
T = np.fft.fft(t)
frequencies = np.fft.fftfreq(len(t), d=1)

# 绘制原始信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t)
plt.title('Original Signal')

# 绘制幅度谱
plt.subplot(2, 1, 2)
plt.stem(frequencies, np.abs(T), 'b', markerfmt=" ", basefmt="-b")
plt.title('Amplitude Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()