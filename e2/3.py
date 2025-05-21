import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 2/experm02_wheatEar.png')

def log_transform(x):
    return np.log(1 + x * 255 / 256) / np.log(1 + 255)

# 应用对数变换
log_image = np.vectorize(log_transform)(image)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Log Transformed Image')
plt.imshow(log_image, cmap='gray')
plt.show()