import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
wheat_image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 2/experm02_wheatEar.png', cv2.IMREAD_GRAYSCALE)

# 直方图均衡化
equalized_wheat = cv2.equalizeHist(wheat_image)

# 显示图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(wheat_image, cmap='gray')
plt.title('Original Wheat Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_wheat, cmap='gray')
plt.title('Equalized Wheat Image')
plt.axis('off')

plt.show()

# 显示直方图
plt.figure()
plt.hist(wheat_image.ravel(), 256, [0, 256], alpha=0.5, label='Original')
plt.hist(equalized_wheat.ravel(), 256, [0, 256], alpha=0.5, label='Equalized')
plt.legend()
plt.title('Histogram Comparison')
plt.show()
