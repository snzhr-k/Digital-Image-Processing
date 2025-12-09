
import cv2
import numpy as np

noise_salt_percentage = 0.05
noise_pepper_percentage = 0.05
median_size = 3
blur_size = 3


def on_median_size_change(pos):
    global median_size

    median_size = pos * 2 + 3
    print('Median size:', median_size)
    do_filter()


def on_blur_size_change(pos):
    global blur_size

    blur_size = pos * 2 + 3
    print('Blur size: ' + str(blur_size) + 'x' + str(blur_size))
    do_filter()


def on_salt_change(pos):
    global im, im_noise
    global noise_salt_percentage, noise_pepper_percentage

    noise_salt_percentage = pos / 100.0
    im_noise = add_salt_and_pepper_noise(im, noise_salt_percentage, noise_pepper_percentage)
    cv2.imshow('im_noise', im_noise)
    do_filter()


def on_pepper_change(pos):
    global im, im_noise
    global noise_salt_percentage, noise_pepper_percentage

    noise_pepper_percentage = pos / 100.0
    im_noise = add_salt_and_pepper_noise(im, noise_salt_percentage, noise_pepper_percentage)
    cv2.imshow('noise', im_noise)
    do_filter()


def do_filter():
    global median_size, blur_size
    global im_noise

    im_median = cv2.medianBlur(im_noise, median_size)
    cv2.imshow('median', im_median)

    im_noise_blur = cv2.blur(im_noise, (blur_size, blur_size))
    cv2.imshow('blur', im_noise_blur)


def add_point_noise(im_in, percentage, value):
    im_noise_res = np.copy(im_in)
    n = int(im_in.shape[0] * im_in.shape[1] * percentage)
    # print(n)

    for k in range(1, n):
        i = np.random.randint(0, im_in.shape[1])
        j = np.random.randint(0, im_in.shape[0])

        if im_in.ndim == 2:
            im_noise_res[j, i] = value

        if im_in.ndim == 3:
            im_noise_res[j, i] = [value, value, value]

    return im_noise_res


def add_salt_and_pepper_noise(im_in, percentage1, percentage2):
    n = add_point_noise(im_in, percentage1, 255)   # Só
    n2 = add_point_noise(n, percentage2, 0)         # Bors

    return n2


# im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('screen01_h.png', cv2.IMREAD_GRAYSCALE)

im_noise = add_salt_and_pepper_noise(im, noise_salt_percentage, noise_pepper_percentage)
cv2.imshow('noise', im)
cv2.createTrackbar('salt', 'noise', 5, 25, on_salt_change)
cv2.createTrackbar('pepper', 'noise', 5, 25, on_pepper_change)

cv2.imshow('median', im_noise)
cv2.createTrackbar('size', 'median', 0, 20, on_median_size_change)
on_median_size_change(0)

cv2.imshow('blur', im_noise)
cv2.createTrackbar('size', 'blur', 0, 20, on_blur_size_change)
on_blur_size_change(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
