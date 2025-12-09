
import cv2
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('screen01_h.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('Sudoku_h.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('FrenchCardsShapes.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('histogram/Picture5.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('Tulips_01_800.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('img', im)
print('Min: {}'.format(np.min(im)))
print('Max: {}'.format(np.max(im)))

# Histogram with 256 bins

hist_gray = cv2.calcHist([im], [0], None, [256], [0, 256])
print(hist_gray.shape)

fig = plt.figure(figsize=(4, 2), dpi=100)
# plt.plot(hist_gray)
# plt.scatter(np.arange(256), hist_gray, s=1)
# plt.bar(np.arange(256), np.transpose(hist_gray.astype(int))[0])
plt.vlines(np.arange(256), 0, hist_gray)
# plt.xlim([0, 255])
plt.ylim([0, np.max(hist_gray)])
fig.tight_layout(pad=0)
plt.show()

# Histogram with 16 bins
hist_gray2 = cv2.calcHist([im], [0], None, [16], [0, 256])
print(hist_gray2.shape)

fig2 = plt.figure(figsize=(4, 2), dpi=100)
# plt.plot(hist_gray2)
# plt.scatter(np.arange(16), hist_gray2, s=5)
# plt.bar(np.arange(16), np.transpose(hist_gray2.astype(int))[0])
plt.vlines(np.arange(16), 0, hist_gray2)
# plt.xlim([0, 15])
plt.ylim([0, np.max(hist_gray2)])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
