
import cv2
import numpy as np

MAGNITUDE_THRESH_PERCENT = 0.2

im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('PalPant_800.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_GRAYSCALE)


# Image normalization and display
def display_image(window, im_in):
    disp = cv2.normalize(im_in, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow(window, disp)


Gx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]])

im_dx = cv2.filter2D(im, cv2.CV_32F, Gx)
display_image('Ix', im_dx)

Gy = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]])

im_dy = cv2.filter2D(im, cv2.CV_32F, Gy)
display_image('Iy', im_dy)

im_gradient_magnitude = cv2.magnitude(im_dx, im_dy)
display_image('Gradient magnitude', im_gradient_magnitude)

val_max_magnitude = np.amax(im_gradient_magnitude)
print('Maximal magnitude value =', val_max_magnitude)
val_magnitude_th = val_max_magnitude * MAGNITUDE_THRESH_PERCENT
print('Threshold normalized percentage =', MAGNITUDE_THRESH_PERCENT)
print('Magnitude threshold value =', val_magnitude_th)
_, im_th = cv2.threshold(im_gradient_magnitude, val_magnitude_th, 1.0, cv2.THRESH_BINARY)
display_image('Thresholded gradient magnitude', im_th)

cv2.waitKey(0)
cv2.destroyAllWindows()
