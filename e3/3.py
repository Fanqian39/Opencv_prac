import cv2
import numpy as np
import matplotlib.pyplot as plt

# 数字图像和模板
image = np.array([
    [6, 6, 7, 2, 3],
    [0, 4, 1, 0, 2],
    [3, 5, 5, 3, 8],
    [0, 2, 2, 1, 2],
    [0, 1, 2, 5, 3]
])

template = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 相关运算
def correlation(image, template):
    result = np.zeros((image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.sum(image[i:i+template.shape[0], j:j+template.shape[1]] * template)
    return result

# 卷积运算
def convolution(image, template):
    result = np.zeros((image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1))
    #卷积运算跟相关运算的区别：卷积运算需要对模板进行上下、左右的翻转
    template = np.fliplr(np.flipud(template))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.sum(image[i:i+template.shape[0], j:j+template.shape[1]] * template)
    return result

# 显示运算结果
correlation_result = correlation(image, template)
convolution_result = convolution(image, template)

print("Correlation Result:\n", correlation_result)
print("Convolution Result:\n", convolution_result)