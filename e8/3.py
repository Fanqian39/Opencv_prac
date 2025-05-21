import cv2
import numpy as np
from matplotlib import pyplot as plt


def calculate_glcm(image, distances, angles):
    rows, cols = image.shape
    glcm = np.zeros((256, 256), dtype=np.uint16)

    for row in range(rows):
        for col in range(cols):
            pixel_value = image[row, col]
            for d, angle in zip(distances, angles):
                row_offset = int(d * np.cos(angle))
                col_offset = int(d * np.sin(angle))
                if 0 <= row + row_offset < rows and 0 <= col + col_offset < cols:
                    glcm[pixel_value, image[row + row_offset, col + col_offset]] += 1

    return glcm


def calculate_features(glcm):
    sum_glcm = np.sum(glcm)
    energy = np.sum(glcm ** 2) / sum_glcm
    contrast = np.sum(glcm * np.arange(256)[:, None] ** 2) / sum_glcm
    homogeneity = np.sum(glcm / (1 + np.arange(256)[:, None] ** 2))
    correlation = np.sum(glcm * np.arange(256)[:, None] * np.arange(256)[None, :]) / (
                sum_glcm * np.std(np.arange(256)) ** 2)

    return energy, contrast, homogeneity, correlation


def process_image(image_path):
    img = cv2.imread(image_path, 0)
    glcm = calculate_glcm(img, distances=[1], angles=[0])
    energy, contrast, homogeneity, correlation = calculate_features(glcm)

    print(f"Energy: {energy}, Contrast: {contrast}, Homogeneity: {homogeneity}, Correlation: {correlation}")


# 处理两幅图像
process_image('C:/Users/25705/Desktop/experiment/experiment 8/experm08_sunflowerSeedling.png')
process_image('C:/Users/25705/Desktop/experiment/experiment 8/experm08_sunflowerSquaring.png')