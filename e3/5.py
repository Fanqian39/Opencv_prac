import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 3/experm03_dairyGoatFace2.png', cv2.IMREAD_GRAYSCALE)

# 拉普拉斯算子
laplacian_kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

def laplacian_filter(image):
    return cv2.filter2D(image, -1, laplacian_kernel)

# 锐化
def sharpen(image, laplacian_image):
    return cv2.add(image, laplacian_image)

laplacian_image = laplacian_filter(image)
sharpened_image = sharpen(image, laplacian_image)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(laplacian_image, cmap='gray')
axs[1].set_title('Laplacian Filter')
axs[2].imshow(sharpened_image, cmap='gray')
axs[2].set_title('Sharpened Image')
plt.show()