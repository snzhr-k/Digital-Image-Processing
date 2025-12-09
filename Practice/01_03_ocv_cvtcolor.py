
# OpenCV2 image reading, color space conversion
# OpenCV online documentation: https://docs.opencv.org/4.12.0/de/d25/imgproc_color_conversions.html

# Importing OpenCV package definitions
import cv2

# Reading color image matrix from file
im_bgr = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_COLOR)
print('im_bgr.shape:', im_bgr.shape)
cv2.imshow('im_bgr', im_bgr)

# Grayscale
im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
cv2.imshow('im_gray', im_gray)
print('im_gray.shape:', im_gray.shape)
cv2.waitKey(0)

# Grayscale to "color"
im_gray2bgr = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)
cv2.imshow('im_gray2bgr', im_gray2bgr)
print('im_gray2bgr.shape:', im_gray2bgr.shape)
cv2.waitKey(0)

# Converting to Lab color space
# L: grayscale intensity
# a, b: chromaticity (color information)
im_Lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2Lab)
print('im_Lab.shape:', im_Lab.shape)
cv2.imshow('im_Lab', im_Lab)
cv2.waitKey(0)

# Converting back to BGR
im_bgr2 = cv2.cvtColor(im_Lab, cv2.COLOR_Lab2BGR)
cv2.imshow('im_bgr2', im_bgr2)
cv2.waitKey(0)

cv2.destroyAllWindows()
