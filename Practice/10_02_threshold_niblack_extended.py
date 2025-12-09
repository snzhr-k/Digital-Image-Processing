
# Note: The Niblack function is available in the OpenCV contrib package.
# If you want to use it, install this version of OpenCV!

import cv2

tb_k = 18
tb_k_max = 20
# Computing Niblack k parameter: (tb_k - 10.0) / 10.0
# We get a number in the interval [-1.0, 1.0], to one decimal place precision
# The initial value will be 0.8.

tb_block_size = 18
tb_block_size_max = 30
# Computing Niblack blockSize parameter: 2 * tb_blockSize + 3
# Possible values: 3, 5, 7, ..., 63
# Initial value is 39.

# Patameter r of Sauvola method
tb_r = 128
tb_r_max = 255

tb_types = [cv2.THRESH_BINARY,
            cv2.THRESH_BINARY_INV,
            cv2.THRESH_TRUNC,
            cv2.THRESH_TOZERO,
            cv2.THRESH_TOZERO_INV]
tb_type_strings = ['cv2.THRESH_BINARY',
                   'cv2.THRESH_BINARY_INV',
                   'cv2.THRESH_TRUNC',
                   'cv2.THRESH_TOZERO',
                   'cv2.THRESH_TOZERO_INV']
# 5 types of OpenCV thresholding according to tb_types.
# Initial value is cv2.THRESH_BINARY.
tb_type = 0
tb_type_max = len(tb_types) - 1

tb_methods = [cv2.ximgproc.BINARIZATION_NIBLACK,
              cv2.ximgproc.BINARIZATION_SAUVOLA,
              cv2.ximgproc.BINARIZATION_WOLF,
              cv2.ximgproc.BINARIZATION_NICK]
tb_methods_strings = ['Niblack', 'Sauvola', 'Wolf', 'Nick']
tb_method = 0
tb_method_max = len(tb_methods) - 1


def refresh_niblack_result():
    k = float((tb_k - 10) / 10)

    block_size = 3
    if tb_block_size > 1:
        block_size = 2 * tb_block_size + 3

    threshold_type = tb_types[tb_type]
    print('Niblack parameters: blocksize={} k={} r={} type={} method={}'.format(block_size,
                                                                                k,
                                                                                tb_r,
                                                                                tb_type_strings[tb_type],
                                                                                tb_methods_strings[tb_method]))
    im_dst = cv2.ximgproc.niBlackThreshold(im_src, 255, threshold_type, block_size, k, binarizationMethod=tb_method, r=tb_r)
    cv2.imshow('Niblack', im_dst)
    cv2.imwrite('threshold_niblack_result.png', im_dst)


def on_nb_trackbar_k(track_pos):
    global tb_k
    tb_k = track_pos
    refresh_niblack_result()


def on_nb_trackbar_block_size(track_pos):
    global tb_block_size
    tb_block_size = track_pos
    refresh_niblack_result()


def on_nb_trackbar_r(track_pos):
    global tb_r
    tb_r = track_pos
    refresh_niblack_result()


def on_nb_trackbar_type(track_pos):
    global tb_type
    tb_type = track_pos
    refresh_niblack_result()


def on_nb_trackbar_method(track_pos):
    global tb_method
    tb_method = track_pos
    refresh_niblack_result()


im_src = cv2.imread('screen01_h.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', im_src)
cv2.namedWindow('Niblack')

cv2.createTrackbar('k', 'Niblack', tb_k, tb_k_max, on_nb_trackbar_k)
cv2.createTrackbar('blockSize', 'Niblack', tb_block_size, tb_block_size_max, on_nb_trackbar_block_size)
cv2.createTrackbar('r', 'Niblack', tb_r, tb_r_max, on_nb_trackbar_r)
cv2.createTrackbar('thresType', 'Niblack', tb_type, tb_type_max, on_nb_trackbar_type)
cv2.createTrackbar('localMethod', 'Niblack', tb_method, tb_method_max, on_nb_trackbar_method)

cv2.waitKey(0)
cv2.destroyAllWindows()
