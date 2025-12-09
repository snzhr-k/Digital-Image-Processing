
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Diagram settings
fig = plt.figure(figsize=(6, 6), dpi=80)
# fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Inversion LUT diagram')
fig.canvas.manager.set_window_title('Inversion LUT diagram')
plt.axis('equal')
plt.xlabel('Original intensity value')
plt.ylabel('Result of point operation')
plt.xlim([0, 255])
plt.ylim([0, 255])

# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_GRAYSCALE)
im = cv2.imread('Sudoku_rs.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', im)

# Base points in [0, 255] as 8 bits unsigned integers
x = np.arange(0, 256, 1, np.uint8)

# Inverse using LUT
lut = np.arange(0, 256, 1, np.uint8)
lut = 255 - lut
im_inv = cv2.LUT(im, lut, None)
cv2.imshow('LUT', im_inv)

# Diagram rajzolás
plt.plot(x, x, 'g--', label='Original')
plt.plot(x, lut, 'r-', label='Inverse')
plt.legend()
# plt.savefig('04_01_c_inverse_lut_diagram.png', bbox_inches='tight')
plt.show()

cv2.destroyAllWindows()
