
import cv2
import numpy as np

im_src = cv2.imread('car_numberplate_rs.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', im_src)

im_thresh = np.ndarray(im_src.shape, im_src.dtype)

im_thresh[im_src >= 120] = 255
im_thresh[im_src < 120] = 0
cv2.imshow('Thresholded with value 120', im_thresh)

im_thresh.fill(0)
im_thresh[(im_src >= 80) & (im_src <= 160)] = 255
cv2.imshow('Thresholding the 80-160 interval', im_thresh)

im_clip = im_src.copy()
im_clip[im_src < 80] = 0
im_clip[im_src > 160] = 0
cv2.imshow('Clipping inside an interval', im_clip)
cv2.waitKey(0)

cv2.destroyAllWindows()
