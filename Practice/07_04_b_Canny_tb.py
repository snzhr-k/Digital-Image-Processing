
import cv2
import numpy as np

CANNY_BLUR_KERNEL_SIZE = 5
CANNY_BLUR_SIGMA = 2.0
CANNY_SOBEL_KERNEL_SIZE = 3
CANNY_THRESHOLD_PERCENTAGE_LOWER = 0.1
CANNY_THRESHOLD_PERCENTAGE_HIGHER = 0.2


def get_canny_threshold_values():
    global canny_threshold_tb_1, canny_threshold_tb_2

    if canny_threshold_tb_1 < canny_threshold_tb_2:
        threshold_1, threshold_2 = canny_threshold_tb_1, canny_threshold_tb_2
    else:
        threshold_1, threshold_2 = canny_threshold_tb_2, canny_threshold_tb_1

    return threshold_1, threshold_2  # Values in lower, higher order


def do_canny_threshold():
    global canny_threshold_tb_1, canny_threshold_tb_2
    global im, im_blur

    canny_threshold_1, canny_threshold_2 = get_canny_threshold_values()

    im_edges = cv2.Canny(im_blur, canny_threshold_1, canny_threshold_2, None, CANNY_SOBEL_KERNEL_SIZE, True)
    im_red_edges = cv2.cvtColor(im_blur, cv2.COLOR_GRAY2BGR)
    im_red_edges[im_edges > 0] = [0, 0, 255]
    cv2.imshow('Canny', im_red_edges)


def on_tb_th_1_change(pos):
    global canny_threshold_tb_1

    canny_threshold_tb_1 = pos
    do_canny_threshold()


def on_tb_th_2_change(pos):
    global canny_threshold_tb_2

    canny_threshold_tb_2 = pos
    do_canny_threshold()


def find_max_gradient_value(img_in):
    canny_sobel_kernel_size = CANNY_SOBEL_KERNEL_SIZE

    im_dx = cv2.Sobel(img_in, cv2.CV_32FC1, 1, 0, None, canny_sobel_kernel_size)
    im_dy = cv2.Sobel(img_in, cv2.CV_32FC1, 0, 1, None, canny_sobel_kernel_size)
    im_gradient_magnitude = cv2.magnitude(im_dx, im_dy)
    return int(np.amax(im_gradient_magnitude)) + 1


# Main program

# im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('tree_blur_02.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('PalPant_800.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_GRAYSCALE)
im = cv2.imread('webcam_selfie.jpg', cv2.IMREAD_GRAYSCALE)
im_blur = cv2.GaussianBlur(im, (CANNY_BLUR_KERNEL_SIZE, CANNY_BLUR_KERNEL_SIZE), CANNY_BLUR_SIGMA)

canny_threshold_max = find_max_gradient_value(im_blur)
print('Maximum gradient magnitude value:', canny_threshold_max)
canny_threshold_tb_1 = int(canny_threshold_max * CANNY_THRESHOLD_PERCENTAGE_LOWER)
canny_threshold_tb_2 = int(canny_threshold_max * CANNY_THRESHOLD_PERCENTAGE_HIGHER)

cv2.imshow('Canny', im_blur)  # To make windows size known for trackbars
cv2.createTrackbar('Threshold 1', 'Canny', canny_threshold_tb_1, canny_threshold_max, on_tb_th_1_change)
cv2.createTrackbar('Threshold 2', 'Canny', canny_threshold_tb_2, canny_threshold_max, on_tb_th_2_change)
cv2.waitKey(0)

cv2.destroyAllWindows()
