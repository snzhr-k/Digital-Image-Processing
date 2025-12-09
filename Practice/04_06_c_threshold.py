
import cv2
import numpy as np

# im_src = cv2.imread('screen01_h.png', cv2.IMREAD_GRAYSCALE)
im_src = cv2.imread('Sudoku_h.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Source', im_src)

print('Global threshold at intensity value 128.')
_, im_thresh = cv2.threshold(im_src, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('Result', im_thresh)
cv2.waitKey(0)

threshold, im_thresh = cv2.threshold(im_src, -1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print('Detected Otsu threshold = {}'.format(threshold))
cv2.imshow('Result', im_thresh)
cv2.waitKey(0)

print('Global threshold at intensity value 128 using Numpy array operations.')
im_thresh = np.ndarray(im_src.shape, im_src.dtype)
im_thresh[im_src >= 128] = 255
im_thresh[im_src < 128] = 0
cv2.imshow('Result', im_thresh)
cv2.waitKey(0)

cv2.destroyAllWindows()
