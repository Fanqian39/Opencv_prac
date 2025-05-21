import cv2
import numpy as np

# 读取图像
ibis_image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 7/experm07_Crested Ibis.png', 0)
petal_image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 7/experm07_petal.png', 0)

# 迭代阈值法
def iterative_threshold(img, max_iter=100):
    threshold = 0
    for i in range(max_iter):
        threshold = np.mean(img) * (1 + i / max_iter)
        img[img < threshold] = 0
        img[img >= threshold] = 255
    return threshold

ibis_threshold = iterative_threshold(ibis_image)
petal_threshold = iterative_threshold(petal_image)

# Otsu算法
ibis_otsu, _ = cv2.threshold(ibis_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
petal_otsu, _ = cv2.threshold(petal_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("朱鹮图像迭代阈值:", ibis_threshold)
print("花瓣图像迭代阈值:", petal_threshold)
print("朱鹮图像Otsu阈值:", ibis_otsu)
print("花瓣图像Otsu阈值:", petal_otsu)

# 检查分割阈值是否相同
same_threshold = ibis_threshold == petal_threshold and ibis_otsu == petal_otsu
print("分割阈值是否相同:", same_threshold)