import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
petal2_image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 7/experm07_petal2.png', 0)
zebra_image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 7/experm07_zebra.png', 0)

def apply_sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    return sobel

def apply_prewitt(image):
    prewittx = cv2.filter2D(image, cv2.CV_64F, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
    prewitty = cv2.filter2D(image, cv2.CV_64F, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
    prewitt = cv2.magnitude(prewittx, prewitty)
    return prewitt

def apply_canny(image, low_threshold, high_threshold):
    canny = cv2.Canny(image, low_threshold, high_threshold)
    return canny

def apply_log(image, ksize):
    log = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    return log

# 应用边界检测算子
sobel_petal2 = apply_sobel(petal2_image)
sobel_zebra = apply_sobel(zebra_image)

prewitt_petal2 = apply_prewitt(petal2_image)
prewitt_zebra = apply_prewitt(zebra_image)

canny_petal2 = apply_canny(petal2_image, 100, 200)
canny_zebra = apply_canny(zebra_image, 100, 200)

log_petal2 = apply_log(petal2_image, ksize=3)
log_zebra = apply_log(zebra_image, ksize=3)

# 显示结果
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1), plt.imshow(sobel_petal2, cmap='gray'), plt.title('Sobel Petal2')
plt.subplot(2, 4, 2), plt.imshow(sobel_zebra, cmap='gray'), plt.title('Sobel Zebra')

plt.subplot(2, 4, 3), plt.imshow(prewitt_petal2, cmap='gray'), plt.title('Prewitt Petal2')
plt.subplot(2, 4, 4), plt.imshow(prewitt_zebra, cmap='gray'), plt.title('Prewitt Zebra')

plt.subplot(2, 4, 5), plt.imshow(canny_petal2, cmap='gray'), plt.title('Canny Petal2')
plt.subplot(2, 4, 6), plt.imshow(canny_zebra, cmap='gray'), plt.title('Canny Zebra')

plt.subplot(2, 4, 7), plt.imshow(log_petal2, cmap='gray'), plt.title('LoG Petal2')
plt.subplot(2, 4, 8), plt.imshow(log_zebra, cmap='gray'), plt.title('LoG Zebra')

plt.tight_layout()
plt.show()