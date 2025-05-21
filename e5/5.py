import numpy as np
import matplotlib.pyplot as plt
import cv2

# 读取图像
image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 5/experm05_cornCob.png', 0)

# 离散傅里叶变换
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# 离散余弦变换
cos_transform = cv2.dct(np.float32(image))

# 绘制结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.log(np.abs(dft_shift)), cmap='gray')
plt.title('DFT Magnitude')

plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(cos_transform)), cmap='gray')
plt.title('DCT Magnitude')
plt.show()