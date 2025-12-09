
import cv2
import numpy as np
from matplotlib import pyplot as plt


im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_COLOR)
# im = cv2.imread('screen01_h.png', cv2.IMREAD_COLOR)
# im = cv2.imread('Sudoku_h.jpg', cv2.IMREAD_COLOR)
# im = cv2.imread('Tulips_01_800.jpg', cv2.IMREAD_COLOR)

cv2.imshow('im', im)

hist_b = cv2.calcHist([im], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([im], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([im], [2], None, [256], [0, 256])

img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

# Histogram of red color channel
fig = plt.figure(figsize=(4,2), dpi=100)
plt.vlines(np.arange(256), 0, hist_r, color='r')
# plt.xlim([0, 255])
plt.ylim([0, np.max(hist_r)])
plt.show()

# Histogram of green color channel
# fig.clear()
plt.figure(figsize=(4,2), dpi=100)
plt.vlines(np.arange(256), 0, hist_g, color='g')
# plt.xlim([0, 255])
plt.ylim([0, np.max(hist_g)])
plt.show()

# Histogram of blue color channel
plt.figure(figsize=(4,2), dpi=100)
# plt.plot(hist_b, color='b')
plt.vlines(np.arange(256), 0, hist_b, color='b')
# plt.xlim([0, 255])
plt.ylim([0, np.max(hist_b)])
plt.show()

# RGB histogram
plt.figure(figsize=(4,2), dpi=100)
plt.plot(hist_b, color='b')
plt.plot(hist_g, color='g')
plt.plot(hist_r, color='r')
plt.plot(hist_gray, color='k')
# plt.xlim([0, 255])
plt.ylim([0, max(np.max(hist_b), np.max(hist_g), np.max(hist_r))])
plt.show()

print(len(hist_g[1:255]))

# RGB histogram skipping values of 0 and 255
plt.figure(figsize=(4,2), dpi=100)
plt.plot(hist_b[1:255], color='b')
plt.plot(hist_g[1:255], color='g')
plt.plot(hist_r[1:255], color='r')
plt.plot(hist_gray, color='k')
# plt.xlim([0, 255])
plt.ylim([0, max(np.max(hist_b[1:255]), np.max(hist_g[1:255]), np.max(hist_r[1:255]))])
plt.show()
