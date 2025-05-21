import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 2/experm02_moth.png')

# 分段线性变换
def piecewise_linear_transform(image, a=64, b=128, c=32, d=192, L=255):
    g = np.zeros_like(image, dtype=np.float32)
    g[image < a] = (c / a) * image[image < a]
    g[(image >= a) & (image < b)] = ((d - c) / (b - a)) * (image[(image >= a) & (image < b)] - a) + c
    g[image >= b] = ((L - 1 - d) / (L - 1 - b)) * (image[image >= b] - b) + d
    return np.uint8(g)

# 应用变换
transformed_image = piecewise_linear_transform(image)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Transformed Image')
plt.imshow(transformed_image, cmap='gray')
plt.show()