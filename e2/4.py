import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
wheat_image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 2/experm02_wheatEar.png', cv2.IMREAD_GRAYSCALE)

# 幂律变换
def power_law_transform(image, gamma=1.0):
    # 确保gamma的值不会导致除以零的情况
    if gamma == 0:
        return np.zeros_like(image)
    # 进行幂律变换
    transformed = np.power(image / 255.0, gamma) * 255
    # 确保数据在0到255的范围内
    transformed = np.clip(transformed, 0, 255)
    return transformed.astype(np.uint8)

gamma = 0.4  # 幂律变换参数
power_transformed_wheat = power_law_transform(wheat_image, gamma)

# 显示图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(wheat_image, cmap='gray')
plt.title('Original Wheat Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(power_transformed_wheat, cmap='gray')
plt.title('Power Transformed Wheat Image')
plt.axis('off')

plt.show()
