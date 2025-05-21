import cv2
import numpy as np
from matplotlib import pyplot as plt


def calculate_geometric_features(image_path):
    # 读取图像
    img = cv2.imread(image_path, 0)
    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算几何特征
    for contour in contours:
        area = cv2.contourArea(contour)  # 面积
        perimeter = cv2.arcLength(contour, True)  # 周长
        x, y, w, h = cv2.boundingRect(contour)  # 包围矩形
        aspect_ratio = float(w) / h  # 矩形度
        circularity = 4 * np.pi * (area / (perimeter * perimeter))  # 圆形度
        # 位置和方向
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            orientation = cv2.minAreaRect(contour)[-1]  # 方向

        print(
            f"Area: {area}, Perimeter: {perimeter}, Aspect Ratio: {aspect_ratio}, Circularity: {circularity}, Center: ({cx}, {cy}), Orientation: {orientation}")

    # 显示图像
    plt.imshow(binary, cmap='gray')
    plt.show()

# 对三幅图像分别计算几何特征
calculate_geometric_features('C:/Users/25705/Desktop/experiment/experiment 8/experm08_leaf.png')
calculate_geometric_features('C:/Users/25705/Desktop/experiment/experiment 8/experm08_leaf2.png')
calculate_geometric_features('C:/Users/25705/Desktop/experiment/experiment 8/experm08_leaf3.png')