
import cv2
import numpy as np

im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)

noise = np.zeros(im.shape[:2], np.int16)
cv2.randn(noise, 0.0, 20.0)

im_noise1 = cv2.add(im, noise, dtype=cv2.CV_8UC1)
cv2.imshow('add', im_noise1)

im_noise2 = cv2.add(im, noise, dtype=cv2.CV_16SC1)
im_noise_norm = cv2.normalize(im_noise2, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow('norm', im_noise_norm)

cv2.waitKey(0)

cv2.destroyAllWindows()
