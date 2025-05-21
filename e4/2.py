import cv2
import matplotlib.pyplot as plt

# 读取图像
goat_img = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 4/experm04_dairyGoat.jpg')

# 提取奶山羊区域
roi = goat_img[170:290, 320:500]

# 定义缩放操作
def resize_image(img, scale_w, scale_h):
    new_width = int(img.shape[1] * scale_w)
    new_height = int(img.shape[0] * scale_h)
    print(new_width,new_height)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 应用缩放处理
scales = [(0.5, 0.5), (1, 1/3), (5/4, 7/5)]
for i, (scale_w, scale_h) in enumerate(scales):
    resized_img = resize_image(roi, scale_w, scale_h)

    # 显示结果
    plt.subplot(1, 3, i+1)
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Scale: ({scale_w}, {round(scale_h,2)})")
    plt.axis('off')

plt.show()