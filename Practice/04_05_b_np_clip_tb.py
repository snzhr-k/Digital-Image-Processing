
import cv2
import numpy as np

th_lower = 44
th_upper = 116
th_mode = 0


def th_clip():
    global im_src, th_lower, th_upper, th_mode
    global im_th_clip, out_file_name

    im_th_clip = np.ndarray(im_src.shape, im_src.dtype)

    if th_mode == 0:
        im_th_clip.fill(0)
        im_th_clip[(im_src >= th_lower) & (im_src <= th_upper)] = 255

    if th_mode == 1:
        im_th_clip = im_src.copy()
        im_th_clip[im_src < th_lower] = 0
        im_th_clip[im_src > th_upper] = 0

    cv2.imshow('im_th_clip', im_th_clip)
    out_file_name = 'im_th_clip_{}_{}_{}.png'.format(th_lower, th_upper, th_mode)


def trackbar_lower(th):
    global th_lower

    th_lower = th
    th_clip()


def trackbar_upper(th):
    global th_upper

    th_upper = th
    th_clip()


def trackbar_mode(x):
    global th_mode

    th_mode = x
    th_clip()


im_src = cv2.imread('car_numberplate_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('Sudoku_h.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('coins_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('utility_meter_readings/1313518560104.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('utility_meter_readings/1313518629386.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('utility_meter_readings/1313575791550.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('utility_meter_readings/1313575818222.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('cards__numbers.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('four-kings-playing-cards-b1pk2m.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('FCards_02_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('10_pr_formula.png', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('huszar1.jpg', cv2.IMREAD_GRAYSCALE)
# im_src = cv2.imread('shapes/shapes_10.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Original', im_src)
im_th_clip = im_src.copy()
out_file_name = 'im_th_clip'

th_clip()

cv2.createTrackbar('Lower', 'im_th_clip', th_lower, 255, trackbar_lower)
cv2.createTrackbar('Upper', 'im_th_clip', th_upper, 255, trackbar_upper)
cv2.createTrackbar('Operation mode', 'im_th_clip', th_mode, 1, trackbar_mode)

while True:
    key = cv2.waitKey(0)

    if key == 27 or key == ord('q'):
        break

    if key == ord('s'):
        cv2.imwrite(out_file_name, im_th_clip)


cv2.destroyAllWindows()
