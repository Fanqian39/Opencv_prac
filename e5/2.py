import numpy as np
import matplotlib.pyplot as plt

# 4x4 数字图像
f = np.array([[1, 3, 8, 9],
             [2, 5, 3, 7],
             [4, 6, 2, 6],
             [6, 7, 2, 0]])

# 先进行行傅里叶变换
Frow = np.fft.fft2(f, axes=(0, 1))
# 先进行列傅里叶变换
Fcol = np.fft.fft2(f, axes=(1, 0))

#调整空间布局，使其便于理解
Frow_col = np.fft.fftshift(Frow)
Fcol_row = np.fft.fftshift(Fcol)

# 验证 Frow_col 和 Fcol_row 是否一致
print("Frow_col:\n", Frow_col)
print("Fcol_row:\n", Fcol_row)

# 绘制变换结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(Frow_col), cmap='gray')
plt.title('Frow_col')

plt.subplot(1, 2, 2)
plt.imshow(np.abs(Fcol_row), cmap='gray')
plt.title('Fcol_row')
plt.show()