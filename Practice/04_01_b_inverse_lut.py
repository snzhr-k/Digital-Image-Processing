
import cv2
import numpy as np

# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_GRAYSCALE)
im = cv2.imread('Sudoku_rs.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', im)

# Inverse using lookup table (LUT)
lut = np.arange(0, 256, 1, np.uint8)
lut = 255 - lut
im_inv = cv2.LUT(im, lut)
cv2.imshow('Inverse LUT', im_inv)
cv2.waitKey(0)

cv2.destroyAllWindows()
