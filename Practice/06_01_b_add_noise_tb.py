
import cv2
import numpy as np


def add_additive_noise(sigma_in):
    global im

    noise = np.zeros(im.shape[:2], np.int16)
    cv2.randn(noise, 0.0, sigma_in)
    im_noise1 = cv2.add(im, noise, dtype=cv2.CV_8UC1)
    cv2.imshow('Noisy', im_noise1)


def on_sigma_change(pos):
    add_additive_noise(pos)


im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Noisy', im)
cv2.createTrackbar('sigma', 'Noisy', 50, 100, on_sigma_change)

cv2.waitKey(0)
cv2.destroyAllWindows()
