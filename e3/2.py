import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 3/experm03_peach.png', cv2.IMREAD_GRAYSCALE)

# 邻域平滑
def neighborhood_smoothing(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    return cv2.filter2D(image, -1, kernel)

# 中值滤波
def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

# 显示原始图像
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# 显示滤波后的图像
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(neighborhood_smoothing(image, 3), cmap='gray')
axs[0, 0].set_title('3x3 Neighborhood Smoothing')
axs[0, 1].imshow(neighborhood_smoothing(image, 5), cmap='gray')
axs[0, 1].set_title('5x5 Neighborhood Smoothing')
axs[1, 0].imshow(median_filter(image, 3), cmap='gray')
axs[1, 0].set_title('3x3 Median Filter')
axs[1, 1].imshow(median_filter(image, 5), cmap='gray')
axs[1, 1].set_title('5x5 Median Filter')
plt.show()