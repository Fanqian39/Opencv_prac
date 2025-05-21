import cv2
from matplotlib import pyplot as plt


def error_template_matching(search_image_path, template_path):
    # 读取搜索图像和模板图像
    search_img = cv2.imread(search_image_path)
    template = cv2.imread(template_path, 0)

    # 检查图像是否成功加载
    if search_img is None or template is None:
        print(
            f"Error: Unable to load one or both images. Search image: {search_image_path}, Template image: {template_path}")
        return

    # 获取模板图像的尺寸
    h, w = template.shape

    # 搜索图像的灰度版本
    search_img_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)

    # 模板匹配
    result = cv2.matchTemplate(search_img_gray, template, cv2.TM_CCOEFF_NORMED)

    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 如果匹配度大于某个阈值，则认为找到了匹配
    threshold = 0.8
    if max_val > threshold:
        # 绘制矩形框
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(search_img, top_left, bottom_right, 255, 2)

    # 显示结果
    plt.imshow(cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB))  # 转换为RGB以正确显示
    plt.show()


# 从搜索图像中找出大熊猫面部
error_template_matching('C:/Users/25705/Desktop/experiment/experiment 8/panda.png',
                        'C:/Users/25705/Desktop/experiment/experiment 8/panda_face.png')