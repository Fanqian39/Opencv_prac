import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
farmfield_image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 7/experm07_farmfield.png', 0)

# 边缘检测
edges = cv2.Canny(farmfield_image, 50, 150, apertureSize=3)

# Hough变换
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# 准备一个空图像来绘制直线
line_image = np.zeros_like(farmfield_image)

# 如果检测到直线，只绘制最长的10条
if lines is not None:
    # 根据长度排序直线
    lines = sorted(lines, key=lambda x: np.hypot(x[0][2] - x[0][0], x[0][3] - x[0][1]), reverse=True)
    for i in range(min(10, len(lines))):
        x1, y1, x2, y2 = lines[i][0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 将绘制了直线的图像叠加到原图上
combined_image = cv2.addWeighted(farmfield_image, 0.8, line_image, 1, 0)

# 显示结果
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines (Top 10 longest)')
plt.axis('off')
plt.show()