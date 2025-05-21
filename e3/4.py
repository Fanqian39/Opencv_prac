import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 3/experm03_dairyGoatFace.png', cv2.IMREAD_GRAYSCALE)

# 钝化掩膜
def low_pass_filter(image):
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float32) / 9
    return cv2.filter2D(image, -1, kernel)

# 高提升滤波
def high_boost_filter(image, low_pass_image):
    return image - low_pass_image + 128

# 显示原始图像和锐化滤波后的图像
low_pass_image = low_pass_filter(image)
high_boost_image = high_boost_filter(image, low_pass_image)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(low_pass_image, cmap='gray')
axs[1].set_title('Low Pass Filter')
axs[2].imshow(high_boost_image, cmap='gray')
axs[2].set_title('High Boost Filter')
plt.show()