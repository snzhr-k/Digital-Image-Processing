
import cv2
import numpy as np
import math

MAGNITUDE_THRESH_PERCENT = 0.2
# ARROW_MAX_LENGTH = 200
SHOW_CONCATENATED_RESULT = True


# Image normalization and display
def display_image(window, im_in):
    disp = cv2.normalize(im_in, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow(window, disp)


def gradient_draw_callback(event, x, y, flags, param):
    global im, im_dx, im_dy, im_gradient_magnitude
    global im_dx_norm, im_dy_norm, im_gradient_magnitude_norm
    global im_magnitude_th
    # global ARROW_MAX_LENGTH

    # If we need the angle of the arrow
    angle_rad = math.atan2(im_dy[y, x], im_dx[y, x])
    angle_deg = math.degrees(angle_rad)
    # If we want to maximize the length of the arrow
    # print(math.degrees(angle_rad), im_gradient_magnitude[y, x])
    # arr_length = ARROW_MAX_LENGTH * im_gradient_magnitude[y, x] / np.amax(im_gradient_magnitude)
    # x2 = round(x + arr_length * math.cos(angle_rad))
    # y2 = round(y + arr_length * math.sin(angle_rad))

    # If we draw the actual length of the arrow, that is sufficient.
    x2 = round(x + im_dx[y, x])
    y2 = round(y + im_dy[y, x])

    img_arrow = cv2.merge([im.copy(), im.copy(), im.copy()])
    cv2.line(img_arrow, (x, y), (x2, y), (0, 255, 0), 2)
    cv2.line(img_arrow, (x2, y), (x2, y2), (255, 0, 0), 2)
    cv2.arrowedLine(img_arrow, (x, y), (x2, y2), (0, 0, 255), 3, tipLength=0.25)
    angle_deg_rounded = round(angle_deg, 2)
    cv2.putText(img_arrow, 'Angle: ' + str(angle_deg_rounded) + ' degrees', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(img_arrow, 'Angle: ' + str(angle_deg_rounded) + ' degrees', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    if not SHOW_CONCATENATED_RESULT:
        cv2.imshow('img', img_arrow)

    im_dx_bgr = cv2.cvtColor(im_dx_norm, cv2.COLOR_GRAY2BGR)
    cv2.circle(im_dx_bgr, (x, y), 2, (32, 32, 32), 4)
    cv2.circle(im_dx_bgr, (x, y), 1, (0, 255, 0), 2)
    cv2.putText(im_dx_bgr, 'Gradient X value: ' + str(im_dx[y, x]), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(im_dx_bgr, 'Gradient X value: ' + str(im_dx[y, x]), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    if not SHOW_CONCATENATED_RESULT:
        cv2.imshow('Ix', im_dx_bgr)

    im_dy_bgr = cv2.cvtColor(im_dy_norm, cv2.COLOR_GRAY2BGR)
    cv2.circle(im_dy_bgr, (x, y), 2, (32, 32, 32), 4)
    cv2.circle(im_dy_bgr, (x, y), 1, (255, 0, 0), 2)
    cv2.putText(im_dy_bgr, 'Gradient Y value: ' + str(im_dy[y, x]), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(im_dy_bgr, 'Gradient Y value: ' + str(im_dy[y, x]), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    if not SHOW_CONCATENATED_RESULT:
        cv2.imshow('Iy', im_dy_bgr)

    im_magnitude_bgr = cv2.cvtColor(im_gradient_magnitude_norm, cv2.COLOR_GRAY2BGR)
    cv2.circle(im_magnitude_bgr, (x, y), 2, (32, 32, 32), 4)
    cv2.circle(im_magnitude_bgr, (x, y), 1, (0, 0, 255), 2)
    magnitude_rounded = round(float(im_gradient_magnitude[y, x]), 2)
    cv2.putText(im_magnitude_bgr, 'Gradient magnitude: ' + str(magnitude_rounded), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(im_magnitude_bgr, 'Gradient magnitude: ' + str(magnitude_rounded), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    if not SHOW_CONCATENATED_RESULT:
        cv2.imshow('Gradient magnitude', im_magnitude_bgr)

    if SHOW_CONCATENATED_RESULT:
        cv2.drawMarker(img_arrow, (x + 1, y + 1), (0, 0, 0), cv2.MARKER_CROSS, 50, 1)
        cv2.drawMarker(img_arrow, (x, y), (192, 192, 192), cv2.MARKER_CROSS, 50, 1)
        img_concat = cv2.vconcat([
                cv2.hconcat([im_dx_bgr, im_dy_bgr]),
                cv2.hconcat([im_magnitude_bgr, img_arrow])
        ])
        cv2.imshow('Gradient visualization', img_concat)

    im_magnitude_th_bgr = cv2.cvtColor(im_magnitude_th, cv2.COLOR_GRAY2BGR)
    cv2.circle(im_magnitude_th_bgr, (x, y), 2, (32, 32, 32), 4)
    cv2.circle(im_magnitude_th_bgr, (x, y), 1, (0, 0, 255), 2)
    display_image('Thresholded gradient magnitude', im_magnitude_th_bgr)


# im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
im = cv2.imread('tree_blur_02.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('PalPant_800.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('im', im)

# ksize = -1   # Scharr kernel
ksize = 3    # 3x3 Sobel
# ksize = 5    # 5x5 Sobel

im_dx = cv2.Sobel(im, cv2.CV_32FC1, 1, 0, None, ksize)
im_dx_norm = cv2.normalize(im_dx, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
if not SHOW_CONCATENATED_RESULT:
    cv2.imshow('Ix', im_dx_norm)

im_dy = cv2.Sobel(im, cv2.CV_32FC1, 0, 1, None, ksize)
im_dy_norm = cv2.normalize(im_dy, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
if not SHOW_CONCATENATED_RESULT:
    cv2.imshow('Iy', im_dy_norm)

im_gradient_magnitude = cv2.magnitude(im_dx, im_dy)
im_gradient_magnitude_norm = cv2.normalize(im_gradient_magnitude, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
if not SHOW_CONCATENATED_RESULT:
    cv2.imshow('Gradient magnitude', im_gradient_magnitude_norm)

val_magnitude_th = round(np.amax(im_gradient_magnitude) * MAGNITUDE_THRESH_PERCENT)
print('Gradient magnitude threshold value =', val_magnitude_th)
_, im_magnitude_th = cv2.threshold(im_gradient_magnitude, val_magnitude_th, 255.0, cv2.THRESH_BINARY)
display_image('Thresholded gradient magnitude', im_magnitude_th)

cv2.setMouseCallback('im', gradient_draw_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()
