import cv2
import numpy as np
from matplotlib import pyplot as plt

def harris_corner_detection(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Harris 角点检测
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # 阈值化
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    # 显示图像
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# 对两幅图像进行Harris角点检测
harris_corner_detection('C:/Users/25705/Desktop/experiment/experiment 8/experm08_leaf.png')
harris_corner_detection('C:/Users/25705/Desktop/experiment/experiment 8/experm08_butterfly.png')