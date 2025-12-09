
import cv2
import numpy as np


def on_threshold_trackbar(pos):
    global im_thresh

    print('Global threshold at intensity value {}.'.format(pos))
    threshold, im_thresh = cv2.threshold(im_src, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow('Result', im_thresh)


im_src = cv2.imread('screen01_h.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Source', im_src)

otsuThreshold, im_thresh = cv2.threshold(im_src, -1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print('Detected Otsu threshold = {}'.format(otsuThreshold))
cv2.imshow('Result', im_thresh)

cv2.createTrackbar('threshold', 'Result', int(otsuThreshold), 255, on_threshold_trackbar)
cv2.waitKey(0)

cv2.destroyAllWindows()
