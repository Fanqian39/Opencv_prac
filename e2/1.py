import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 2/experm02_dairyCow.png', cv2.IMREAD_GRAYSCALE)

inverted_image = 255 - image

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Inverted Image')
plt.imshow(inverted_image, cmap='gray')
plt.show()