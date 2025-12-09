
import cv2
import numpy as np

im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)

noise = np.zeros(im.shape, np.int16)
cv2.randn(noise, 0.0, 20.0)

im_noise = cv2.add(im, noise, dtype=cv2.CV_8UC1)
cv2.imshow('noise', im_noise)

im_noise_blur_5x5 = cv2.blur(im_noise, (5, 5))
cv2.imshow('blur 5x5', im_noise_blur_5x5)

im_noise_gauss_5x5 = cv2.GaussianBlur(im_noise, (5, 5), sigmaX=2.0, sigmaY=2.0)
cv2.imshow('gauss 5x5', im_noise_gauss_5x5)
cv2.waitKey(0)

cv2.destroyAllWindows()
