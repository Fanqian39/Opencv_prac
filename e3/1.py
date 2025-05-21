import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('C:/Users/25705/Desktop/experiment/experiment 3/experm03_leaf.png', cv2.IMREAD_GRAYSCALE)
image = np.float32(image) / 255

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, var=0.01):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss
    return np.clip(noisy, 0, 1)

# 添加椒盐噪声
def add_salt_and_pepper_noise(image, prob=0.05):
    s_vs_p = 0.5
    amount = 0.05
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords[0], coords[1]] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords[0], coords[1]] = 0
    return out

# 添加量化噪声
def add_quantization_noise(image, bits=8):
    scale = 2 ** bits
    return (image * scale).round() / scale

# 显示原始图像和添加噪声后的图像
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 1].imshow(add_gaussian_noise(image), cmap='gray')
axs[0, 1].set_title('Gaussian Noise')
axs[1, 0].imshow(add_salt_and_pepper_noise(image), cmap='gray')
axs[1, 0].set_title('Salt and Pepper Noise')
axs[1, 1].imshow(add_quantization_noise(image), cmap='gray')
axs[1, 1].set_title('Quantization Noise')
plt.show()