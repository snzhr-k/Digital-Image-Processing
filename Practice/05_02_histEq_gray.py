
import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_histogram(src):
    hist_gray = cv2.calcHist([src], [0], None, [256], [0, 256])
    plt.figure(figsize=(4, 2), dpi=100)
    # plt.plot(hist_gray)
    # plt.scatter(np.arange(256), hist_gray, s=1)
    # plt.bar(np.arange(256), np.transpose(hist_gray.astype(int))[0])
    plt.vlines(np.arange(256), 0, hist_gray)
    # plt.xlim([0, 255])
    plt.ylim([0, np.max(hist_gray)])
    plt.show()


# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('screen01_h.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('Sudoku_h.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('IMG_20170402_202233.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('PalPant_800.jpg', cv2.IMREAD_GRAYSCALE)
im = cv2.imread('Olympos_800.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('im', im)
draw_histogram(im)

img_histeq = cv2.equalizeHist(im)
cv2.imshow('im', img_histeq)
draw_histogram(img_histeq)
