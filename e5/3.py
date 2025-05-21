import numpy as np
import matplotlib.pyplot as plt
import cv2

# 读取图像
image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 5/experm05_pumpkin.png', 0)

# 标准傅里叶变换
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 计算幅度谱并进行对数变换以增强可视化效果
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# 高通滤波器设计
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0  # 中心区域设置为0，外围为1

# 应用高通滤波器
fshift = dft_shift * mask

# 逆傅里叶变换
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# 绘制原始图像、标准傅里叶变换后的图像和高通滤波后的图像
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_back, cmap='gray')
plt.title('High Pass Filtered Image')
plt.axis('off')

plt.show()