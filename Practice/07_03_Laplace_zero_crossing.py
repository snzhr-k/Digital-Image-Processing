import cv2
import numpy as np

# Maximal distance from zero to be detected as sign change.
# Image content dependent value!
log_diff = 10

im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('tree_blur_02.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('PalPant_800.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_GRAYSCALE)

im_blur = cv2.GaussianBlur(im, (5, 5), 2.0)
im_LoG = cv2.Laplacian(im_blur, cv2.CV_16S, ksize=3)

# Incorrect solution in OpenCV documentation

im_LoG_abs = cv2.convertScaleAbs(im_LoG)
cv2.imshow('im_LoG_abs', im_LoG_abs)

# Sign change approximation: minimum is negative and maximum is positive in the local neighborhood

im_LoG_min = cv2.morphologyEx(im_LoG, cv2.MORPH_ERODE, np.ones((3, 3)))
im_LoG_max = cv2.morphologyEx(im_LoG, cv2.MORPH_DILATE, np.ones((3, 3)))

im_zero_cross = np.logical_and(im_LoG_min < -log_diff, im_LoG_max > log_diff)
im_zero_cross_th = np.zeros(im_zero_cross.shape, np.uint8)
im_zero_cross_th[im_zero_cross] = 255
cv2.imshow('im_zero_cross_th', im_zero_cross_th)

cv2.waitKey(0)
cv2.destroyAllWindows()
