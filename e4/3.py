import cv2
import matplotlib.pyplot as plt

# 读取图像
cow_img = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 4/experm04_dairyCow.jpg')

# 提取奶牛区域
cow_roi = cow_img[580:730, 450:630]

# 定义旋转操作
def rotate_image(img, angle, scale=1):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (int(w*scale), int(h*scale)))

# 创建一个与原图同样大小的图像，用于输出
output_img = cow_img.copy()

# 旋转角度和缩放比例
operations = [
    (10, 1),         # 旋转10度，不缩放
    (-30, 1),        # 旋转-30度，不缩放
    (10, 1.25)       # 旋转10度，缩放1/4
]

# 三个位置的坐标（这里以原图的四个角为例）
positions = [
    (0, 0),         # 第一个位置
    (450, 0),       # 第二个位置
    (0, 580)]    # 第三个位置

# 应用旋转和缩放操作并将结果放置在不同位置
for i, (angle, scale) in enumerate(operations):
    transformed_img = rotate_image(cow_roi, angle, scale)
    new_width, new_height = transformed_img.shape[1], transformed_img.shape[0]
    start_x, start_y = positions[i]
    # 确保变换后的图像不会超出原图边界
    end_x = min(start_x + new_width, output_img.shape[1])
    end_y = min(start_y + new_height, output_img.shape[0])
    output_img[start_y:end_y, start_x:end_x] = transformed_img[:end_y-start_y, :end_x-start_x]

# 显示结果
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()