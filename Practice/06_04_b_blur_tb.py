
import cv2
import numpy as np

blur_size = 3
gauss_blur_size = 3
gauss_blur_sigma = 2.0
median_size = 3


def add_additive_noise(sigma_in):
    global im, im_noise

    noise = np.zeros(im.shape[:2], np.int16)
    cv2.randn(noise, 0.0, sigma_in)
    im_noise = cv2.add(im, noise, dtype=cv2.CV_8UC1)
    cv2.imshow('im_noise', im_noise)

    do_blur()


def on_noise_sigma_change(pos):
    print('Noise sigma:', pos)
    add_additive_noise(pos)


def on_blur_size_change(pos):
    global blur_size

    blur_size = pos * 2 + 3
    print('Blur size: ' + str(blur_size) + 'x' + str(blur_size))
    do_blur()


def on_gauss_blur_size_change(pos):
    global gauss_blur_size

    gauss_blur_size = pos * 2 + 3
    print('Gauss blur size: ' + str(gauss_blur_size) + 'x' + str(gauss_blur_size))
    do_blur()


def on_gauss_blur_sigma_change(pos):
    global gauss_blur_sigma

    gauss_blur_sigma = 1.0 + pos / 10.0
    print('Gauss blur sigma:', gauss_blur_sigma)
    do_blur()


def on_median_size_change(pos):
    global median_size

    median_size = pos * 2 + 3
    print('Median size:', median_size)
    do_blur()


def do_blur():
    global im_noise
    global blur_size
    global gauss_blur_size, gauss_blur_sigma

    img_noise_blur = cv2.blur(im_noise, (blur_size, blur_size))
    cv2.imshow('blur', img_noise_blur)

    img_noise_gauss_blur = cv2.GaussianBlur(im_noise, (gauss_blur_size, gauss_blur_size),
                                            sigmaX=gauss_blur_sigma, sigmaY=gauss_blur_sigma)
    cv2.imshow('gauss blur', img_noise_gauss_blur)

    img_median = cv2.medianBlur(im_noise, median_size)
    cv2.imshow('median', img_median)


im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)

im_noise = im.copy()
cv2.imshow('noise', im_noise)
cv2.createTrackbar('sigma', 'noise', 20, 100, on_noise_sigma_change)

cv2.imshow('blur', im_noise)
cv2.createTrackbar('size', 'blur', 0, 20, on_blur_size_change)
on_blur_size_change(0)

cv2.imshow('gauss blur', im_noise)
cv2.createTrackbar('size', 'gauss blur', 0, 20, on_gauss_blur_size_change)
on_gauss_blur_size_change(0)
cv2.createTrackbar('sigma', 'gauss blur', 20, 100, on_gauss_blur_sigma_change)
# on_gauss_blur_sigma_change(20)

cv2.imshow('median', im_noise)
cv2.createTrackbar('size', 'median', 0, 20, on_median_size_change)
on_median_size_change(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
