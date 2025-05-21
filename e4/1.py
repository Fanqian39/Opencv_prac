import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
butterfly_img = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 4/experm04_butterfly.jpg')
print(butterfly_img.shape)
# 提取蝴蝶区域
butterfly_roi = butterfly_img[150:340, 300:510].copy()

# 定义左右镜像变换
def horizontal_mirror(img):
    return cv2.flip(img, 1)

# 定义上下镜像变换
def vertical_mirror(img):
    return cv2.flip(img, 0)

# 定义邻域平滑
def neighborhood_smoothing(img):
    return cv2.blur(img, (5, 5))

# 创建一个与原图同样大小的图像，用于输出
output_img = butterfly_img.copy()
print(output_img.shape)

# 四个位置的坐标
positions = [(0, 0), (510, 0),  (0, 340), (510, 340)]

# 应用变换并将结果放置在不同位置
output_img[positions[0][1]:positions[0][1]+190, positions[0][0]:positions[0][0]+210] = butterfly_roi
output_img[positions[1][1]:positions[1][1]+190, positions[1][0]:positions[1][0]+210] = horizontal_mirror(butterfly_roi)
output_img[positions[2][1]:positions[2][1]+190, positions[2][0]:positions[2][0]+210] = neighborhood_smoothing(butterfly_roi)
output_img[positions[3][1]:positions[3][1]+190, positions[3][0]:positions[3][0]+210] = vertical_mirror(butterfly_roi)

# 显示结果
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()