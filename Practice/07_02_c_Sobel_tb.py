
import cv2
import numpy as np

MAGNITUDE_THRESH_PERCENT = 0.2

im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('tree_blur_02.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('PalPant_800.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def on_tb_th_change(pos):
    global im_gradient_magnitude, img_bgr
    global val_magnitude_th

    _, img_th = cv2.threshold(im_gradient_magnitude, pos, 1.0, cv2.THRESH_BINARY)
    # display_image('Thresholded gradient magnitude', img_th)
    img_bgr_edge = img_bgr.copy()
    img_bgr_edge[img_th > 0] = [0, 0, 255]
    cv2.imshow('Thresholded gradient magnitude', img_bgr_edge)


# Image normalization and display
def display_image(window, im_in):
    disp = cv2.normalize(im_in, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow(window, disp)


# ksize = -1   # Scharr kernel
ksize = 3    # 3x3 Sobel
# ksize = 5    # 5x5 Sobel

im_dx = cv2.Sobel(im, cv2.CV_32FC1, 1, 0, None, ksize)
display_image('Ix', im_dx)

im_dy = cv2.Sobel(im, cv2.CV_32FC1, 0, 1, None, ksize)
display_image('Iy', im_dy)

im_gradient_magnitude = cv2.magnitude(im_dx, im_dy)
im_gradient_magnitude_norm = cv2.normalize(im_gradient_magnitude, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
display_image('Gradient magnitude', im_gradient_magnitude)

val_max_magnitude = np.amax(im_gradient_magnitude)
print('Maximal magnitude value =', val_max_magnitude)
val_magnitude_th = round(val_max_magnitude * MAGNITUDE_THRESH_PERCENT)
print('Initial magnitude threshold value =', val_magnitude_th)

cv2.imshow('Thresholded gradient magnitude', im_gradient_magnitude_norm)
cv2.createTrackbar('Magnitude threshold', 'Thresholded gradient magnitude', val_magnitude_th, round(val_max_magnitude + 1), on_tb_th_change)

cv2.waitKey(0)
cv2.destroyAllWindows()
