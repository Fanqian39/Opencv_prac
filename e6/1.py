import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 6/experm06_strawberry.png', 0)  # 0 表示以灰度模式读取

# 定义3x3和7x7的结构元素
kernel_3x3 = np.ones((3,3), np.uint8)
kernel_7x7 = np.ones((7,7), np.uint8)

# 1. 腐蚀操作
eroded_3x3 = cv2.erode(image, kernel_3x3, iterations=1)
eroded_7x7 = cv2.erode(image, kernel_7x7, iterations=1)

# 2. 膨胀操作
dilated_3x3 = cv2.dilate(image, kernel_3x3, iterations=1)
dilated_7x7 = cv2.dilate(image, kernel_7x7, iterations=1)

# 3. 开运算（先腐蚀后膨胀）
opened_3x3 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_3x3)
opened_7x7 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_7x7)

# 4. 闭运算（先膨胀后腐蚀）
closed_3x3 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_3x3)
closed_7x7 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_7x7)

# 显示结果
plt.figure(figsize=(12, 12))

plt.subplot(2, 4, 1), plt.imshow(eroded_3x3, cmap='gray'), plt.title('Eroded 3x3')
plt.subplot(2, 4, 2), plt.imshow(eroded_7x7, cmap='gray'), plt.title('Eroded 7x7')
plt.subplot(2, 4, 3), plt.imshow(dilated_3x3, cmap='gray'), plt.title('Dilated 3x3')
plt.subplot(2, 4, 4), plt.imshow(dilated_7x7, cmap='gray'), plt.title('Dilated 7x7')
plt.subplot(2, 4, 5), plt.imshow(opened_3x3, cmap='gray'), plt.title('Opened 3x3')
plt.subplot(2, 4, 6), plt.imshow(opened_7x7, cmap='gray'), plt.title('Opened 7x7')
plt.subplot(2, 4, 7), plt.imshow(closed_3x3, cmap='gray'), plt.title('Closed 3x3')
plt.subplot(2, 4, 8), plt.imshow(closed_7x7, cmap='gray'), plt.title('Closed 7x7')

plt.show()