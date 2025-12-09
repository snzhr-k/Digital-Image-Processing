
import cv2
import numpy as np


# HSV interval segmentation
def hsv_segment(interval_h, interval_s, interval_v, wndtitle):
    global im_hsv

    arr_min_hsv = np.array([interval_h[0], interval_s[0], interval_v[0]])
    arr_max_hsv = np.array([interval_h[1], interval_s[1], interval_v[1]])
    im_segmented = cv2.inRange(im_hsv, arr_min_hsv, arr_max_hsv)
    cv2.imshow(wndtitle, im_segmented)


# Main program

im = cv2.imread('fruits_h.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Input', im)

im_blur = cv2.GaussianBlur(im, (5, 5), sigmaX=2.0, sigmaY=2.0)
im_hsv = cv2.cvtColor(im_blur, cv2.COLOR_BGR2HSV)

# Oranges
hsv_segment((10, 20), (205, 255), (155, 255), 'Narancs')

# Lemon
hsv_segment((20, 30), (160, 255), (175, 255), 'Citrom')

# Pomelo
hsv_segment((20, 55), (70, 255), (60, 150), 'Pomelo')

cv2.waitKey(0)
cv2.destroyAllWindows()
