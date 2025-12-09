
import cv2
import numpy as np
import math

# Size of the result image in pixels
HSV_SIZE_X = 256
HSV_SIZE_Y = HSV_SIZE_X
# Values to work with
HSV_SIZE_HALF = HSV_SIZE_X >> 1
HSV_CENTER_X = HSV_SIZE_HALF
HSV_CENTER_Y = HSV_SIZE_HALF
# Make sure the maximal distance of the circle gets value 255
HSV_FACTOR = 255 / HSV_SIZE_HALF
# Default V value
HSV_V_VALUE = 220


def on_trackbar_change(x):
    global hsv_palette_circle, palette_bgr, hsv_palette_circle_mask

    hsv_palette_circle[hsv_palette_circle_mask > 0] = x
    palette_bgr = cv2.cvtColor(hsv_palette_circle, cv2.COLOR_HSV2BGR)
    cv2.imshow('HSV Circle', palette_bgr)


def mouse_event(event, x, y, flags, param):
    global hsv_palette_circle, palette_bgr, hsv_palette_circle_mask, HSV_CENTER_X, HSV_CENTER_Y, HSV_SIZE_HALF

    if event == cv2.EVENT_LBUTTONDOWN and hsv_palette_circle_mask[y, x, 2] > 0:
        print('H:', hsv_palette_circle[y, x, 0], 'S:', hsv_palette_circle[y, x, 1], 'V:', hsv_palette_circle[y, x, 2])
        im_overlay = palette_bgr.copy()
        cv2.line(im_overlay, (HSV_CENTER_X, HSV_CENTER_Y), (HSV_CENTER_X + HSV_SIZE_HALF, HSV_CENTER_Y), (0, 0, 0), 1)
        cv2.line(im_overlay, (HSV_CENTER_X, HSV_CENTER_Y), (x, y), (255, 255, 255), 1)
        cv2.circle(im_overlay, (x, y), 3, (255, 255, 255), -1)
        cv2.imshow('HSV Circle', im_overlay)
        cv2.waitKey(5000)
        cv2.imshow('HSV Circle', palette_bgr)


hsv_palette_circle = np.ndarray((HSV_SIZE_Y, HSV_SIZE_X, 3), np.uint8)
hsv_palette_circle_mask = np.ndarray((HSV_SIZE_Y, HSV_SIZE_X, 3), np.uint8)

for j in range(0, HSV_SIZE_Y):
    for i in range(0, HSV_SIZE_X):
        dist = math.sqrt((j - HSV_CENTER_Y) ** 2 + (i - HSV_CENTER_X) ** 2)
        if dist >= HSV_SIZE_X / 2:
            hsv_palette_circle[j, i] = [0, 0, 255]
            hsv_palette_circle_mask[j, i] = [0, 0, 0]
        else:
            hsv_palette_circle_mask[j, i] = [0, 0, 255]
            hsv_palette_circle[j, i, 2] = HSV_V_VALUE
            hsv_palette_circle[j, i, 1] = dist * HSV_FACTOR
            angle = math.atan2((HSV_SIZE_Y - j - HSV_CENTER_Y), (i - HSV_CENTER_X)) / 2
            if angle < 0:
                hsv_palette_circle[j, i, 0] = math.degrees(angle) + 180
            else:
                hsv_palette_circle[j, i, 0] = math.degrees(angle)

palette_bgr = cv2.cvtColor(hsv_palette_circle, cv2.COLOR_HSV2BGR)
cv2.imshow('HSV Circle', palette_bgr)
cv2.imwrite('HSV_circle.png', palette_bgr)

cv2.createTrackbar('V value', 'HSV Circle', HSV_V_VALUE, 255, on_trackbar_change)
cv2.setMouseCallback('HSV Circle', mouse_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
