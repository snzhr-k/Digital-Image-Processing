
import cv2
import numpy as np
import math

last_coord = []
region_selected = False
mouse_clicked = False
actual_pixel_hsv = []
diff_h = 5
diff_s = 50
diff_v = 50

# Values for HSV visualization
# Size of the result image in pixels
HSV_SIZE_X = 256
HSV_SIZE_Y = HSV_SIZE_X
# Geometry values
HSV_SIZE_HALF = HSV_SIZE_X >> 1
HSV_CENTER_X = HSV_SIZE_HALF
HSV_CENTER_Y = HSV_SIZE_HALF
HSV_VALUE_SLIDER_HEIGHT = 30
HSV_V_VALUE_TICK_HEIGHT_HALF = 3
HSV_V_VALUE_TICK_BOUND_HEIGHT_HALF = 5
# To make sure the maximal distance of the circle gets value 255
HSV_FACTOR = 255 / HSV_SIZE_HALF
# Default V value
HSV_DEFAULT_V_VALUE = 192


def draw_hs_line(im_in, hue, sat, color, draw_circle = False):
    angle = hue * 2
    radius = int(float(HSV_SIZE_HALF) * sat / 255.0)
    dest_x = int(HSV_CENTER_X + radius * math.cos(math.radians(angle)))
    dest_y = HSV_SIZE_Y - int(HSV_CENTER_Y + radius * math.sin(math.radians(angle)))
    cv2.line(im_in, (HSV_CENTER_X, HSV_CENTER_Y), (dest_x, dest_y), color, 1)
    if draw_circle:
        cv2.circle(im_in, (dest_x, dest_y), 3, color, -1)


def draw_v_tick(im_in, val, hh, color, thick=1):
    tick_y = HSV_SIZE_Y + (HSV_VALUE_SLIDER_HEIGHT >> 1)
    tick_x = int(float(HSV_SIZE_X) * val / 255.0)
    cv2.line(im_in, (tick_x, tick_y - hh), (tick_x, tick_y + hh), color, thick)


def draw_saturation_circle(im_in, sat, color):
    radius = int(float(HSV_SIZE_HALF) * sat / 255.0)
    cv2.circle(im_in, (HSV_CENTER_X, HSV_CENTER_Y), radius, color)


def update_hsv_palette(value_v):
    global hsv_palette_circle, hsv_palette_circle_mask, hsv_palette_bgr

    hsv_palette_circle[hsv_palette_circle_mask > 0] = value_v
    hsv_palette_bgr = cv2.cvtColor(hsv_palette_circle, cv2.COLOR_HSV2BGR)
    tick_y = HSV_SIZE_Y + (HSV_VALUE_SLIDER_HEIGHT >> 1)
    cv2.line(hsv_palette_bgr, (0, tick_y), (HSV_SIZE_X, tick_y), (0, 0, 0), 1)
    draw_v_tick(hsv_palette_bgr, value_v, HSV_V_VALUE_TICK_HEIGHT_HALF, (0, 0, 0), 3)


def visualize_hsv_segment_parameters(pixel_hsv, min_h, max_h, min_s, max_s, min_v, max_v):
    # HSV visualization
    update_hsv_palette(pixel_hsv[2])
    draw_hs_line(hsv_palette_bgr, pixel_hsv[0], pixel_hsv[1], (0, 0, 0), True)
    draw_hs_line(hsv_palette_bgr, min_h, 255, (255, 255, 255))
    draw_hs_line(hsv_palette_bgr, max_h, 255, (255, 255, 255))
    draw_saturation_circle(hsv_palette_bgr, min_s, (0, 255, 0))
    draw_saturation_circle(hsv_palette_bgr, max_s, (0, 0, 255))
    draw_v_tick(hsv_palette_bgr, min_v, HSV_V_VALUE_TICK_BOUND_HEIGHT_HALF, (0, 192, 0), 3)
    draw_v_tick(hsv_palette_bgr, max_v, HSV_V_VALUE_TICK_BOUND_HEIGHT_HALF, (0, 0, 192), 3)
    draw_v_tick(hsv_palette_bgr, pixel_hsv[2], HSV_V_VALUE_TICK_HEIGHT_HALF, (0, 0, 0), 3)
    if min_h > max_h:
        min_h = min_h - 180
    print('H:', min_h, max_h, '; S:', min_s, max_s, '; V:', min_v, max_v)
    cv2.imshow('palette', hsv_palette_bgr)


def segment_hsv_point():
    # Taking global variables
    global im, im_hsv, hsv_palette_circle
    global last_coord, region_selected, mouse_clicked
    global diff_h, diff_s, diff_v
    global actual_pixel_hsv

    # If there were no previous mouse click
    if len(actual_pixel_hsv) == 0:
        return

    pixel_hsv = actual_pixel_hsv

    # Handling underflow in segmentation
    min_h = pixel_hsv[0] - diff_h

    # Handling overflow in segmentation
    max_h = pixel_hsv[0] + diff_h

    if pixel_hsv[1] > diff_s:
        min_s = pixel_hsv[1] - diff_s
    else:
        min_s = 0

    if pixel_hsv[1] < (255 - diff_s):
        max_s = pixel_hsv[1] + diff_s
    else:
        max_s = 255

    if pixel_hsv[2] > diff_v:
        min_v = pixel_hsv[2] - diff_v
    else:
        min_v = 0

    if pixel_hsv[2] < (255 - diff_v):
        max_v = pixel_hsv[2] + diff_v
    else:
        max_v = 255

    # HSV interval segmentation
    # Handling the under- and overflow of value H: 2 interval segmentation if necessary
    vis_min_h = min_h
    vis_max_h = max_h
    segmented = np.zeros(im_hsv.shape[0:2], np.uint8)
    if min_h < 0:
        while min_h < 0:
            min_h = min_h + 180
        min_hsv = np.array([min_h, min_s, min_v])
        max_hsv = np.array([180, max_s, max_v])
        segmented = cv2.inRange(im_hsv, min_hsv, max_hsv)
        vis_min_h = min_h
        min_h = 0

    if max_h > 180:
        while max_h > 180:
            max_h = max_h - 180
        min_hsv = np.array([0, min_s, min_v])
        max_hsv = np.array([max_h, max_s, max_v])
        segmented_temp = cv2.inRange(im_hsv, min_hsv, max_hsv)
        segmented = cv2.bitwise_or(segmented, segmented_temp)
        vis_max_h = max_h
        max_h = 180

    min_hsv = np.array([min_h, min_s, min_v], np.uint8)
    max_hsv = np.array([max_h, max_s, max_v], np.uint8)
    print(min_hsv, max_hsv)
    segmented_temp = cv2.inRange(im_hsv, min_hsv, max_hsv)
    segmented = cv2.bitwise_or(segmented, segmented_temp)
    cv2.imshow('segmented', segmented)

    # HSV visualization
    visualize_hsv_segment_parameters(pixel_hsv, vis_min_h, vis_max_h, min_s, max_s, min_v, max_v)


def segment_hsv_region(x, y):
    global last_coord, im_hsv, region_selected, hsv_palette_bgr
    global diff_h, diff_s, diff_v, actual_pixel_hsv

    # left-top and right-bottom coordinate of the last two mouse clicks
    minx = min(last_coord[0], x)
    miny = min(last_coord[1], y)
    maxx = max(last_coord[0], x)
    maxy = max(last_coord[1], y)

    # Return if it is not a real rectangle
    if minx != maxx and miny != maxy:
        # Computing HSV min and max in the selected rectangular area
        im_cut = im_hsv[miny:maxy, minx:maxx]
        cut_min_h = np.min(im_cut[:, :, 0])
        cut_max_h = np.max(im_cut[:, :, 0])
        cut_min_s = np.min(im_cut[:, :, 1])
        cut_max_s = np.max(im_cut[:, :, 1])
        cut_min_v = np.min(im_cut[:, :, 2])
        cut_max_v = np.max(im_cut[:, :, 2])
        # Computing the parameters of pixel segmentation
        actual_pixel_hsv = (int((int(cut_max_h) + int(cut_min_h)) / 2), int((int(cut_max_s) + int(cut_min_s)) / 2), int((int(cut_max_v) + int(cut_min_v)) / 2))
        diff_h = (cut_max_h - cut_min_h) >> 1
        diff_s = (cut_max_s - cut_min_s) >> 1
        diff_v = (cut_max_v - cut_min_v) >> 1
        segment_hsv_point()

        # Selection to inital state
        last_coord = []
        region_selected = False


def mouse_click(event, x, y, flags, param):
    # Taking global variables
    global im, im_hsv, hsv_palette_circle, actual_pixel_hsv
    global last_coord, region_selected, mouse_clicked
    global diff_h, diff_s, diff_v

    if event == cv2.EVENT_LBUTTONDOWN:
        last_coord = (x, y)
        region_selected = False
        mouse_clicked = True

    if event == cv2.EVENT_MOUSEMOVE:
        if mouse_clicked:
            region_selected = True
            orig_img_overlay = im.copy()
            cv2.rectangle(orig_img_overlay, last_coord, (x, y), (0, 0, 192), 1)
            cv2.imshow('origImg', orig_img_overlay)

    if event == cv2.EVENT_LBUTTONUP:
        mouse_clicked = False

        if not region_selected:
            # Red crosshair to the clicked location
            orig_img_overlay = im.copy()
            orig_img_overlay[y, :] = [0, 0, 192]
            orig_img_overlay[:, x] = [0, 0, 192]
            cv2.imshow('origImg', orig_img_overlay)
            # Segmentation
            actual_pixel_hsv = im_hsv[y, x]
            segment_hsv_point()

        if region_selected:
            # The region rectangle has already been drawn, not necessary here
            # Computing segmentation parameters in the selected region
            # lastCoord gives the other corner coordinate
            segment_hsv_region(x, y)


# Main program

im = cv2.imread('fruits.jpg', cv2.IMREAD_COLOR)
# im = cv2.imread('fruits_h.jpg', cv2.IMREAD_COLOR)
# im = cv2.imread('car_numberplate_rs.jpg', cv2.IMREAD_COLOR)
# im = cv2.imread('butterfly.jpg', cv2.IMREAD_COLOR)
# im = cv2.imread('PalPant_800.jpg', cv2.IMREAD_COLOR)

im = cv2.GaussianBlur(im, (5, 5), sigmaX=2.0, sigmaY=2.0)
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

hsv_palette_circle = np.ndarray((HSV_SIZE_Y + HSV_VALUE_SLIDER_HEIGHT, HSV_SIZE_X, 3), np.uint8)
hsv_palette_circle_mask = np.ndarray((HSV_SIZE_Y + HSV_VALUE_SLIDER_HEIGHT, HSV_SIZE_X, 3), np.uint8)

print('Computing HSV palette image...')
for j in range(0, HSV_SIZE_Y + HSV_VALUE_SLIDER_HEIGHT):
    for i in range(0, HSV_SIZE_X):
        dist = math.sqrt((j - HSV_CENTER_Y) ** 2 + (i - HSV_CENTER_X) ** 2)
        if dist >= HSV_SIZE_X / 2:
            hsv_palette_circle[j, i] = [0, 0, 255]
            hsv_palette_circle_mask[j, i] = [0, 0, 0]
        else:
            hsv_palette_circle_mask[j, i] = [0, 0, 255]
            hsv_palette_circle[j, i, 2] = HSV_DEFAULT_V_VALUE
            hsv_palette_circle[j, i, 1] = dist * HSV_FACTOR
            angle = math.atan2((HSV_SIZE_Y - j - HSV_CENTER_Y), (i - HSV_CENTER_X)) / 2
            if angle < 0:
                hsv_palette_circle[j, i, 0] = math.degrees(angle) + 180
            else:
                hsv_palette_circle[j, i, 0] = math.degrees(angle)

print('Computing done.')
print('Usable keys: h, H, s, S, v, V, q')
print('Click a pixel or select a region using left mouse button.')

hsv_palette_bgr = cv2.cvtColor(hsv_palette_circle, cv2.COLOR_HSV2BGR)
update_hsv_palette(HSV_DEFAULT_V_VALUE)
cv2.imshow('palette', hsv_palette_bgr)
cv2.imshow('origImg', im)
cv2.setMouseCallback('origImg', mouse_click)

while True:
    key = cv2.waitKey(0)

    if key == ord('q'):
        break

    if key == ord('h'):
        if diff_h > 1:
            diff_h = diff_h - 1
        segment_hsv_point()

    if key == ord('H'):
        if diff_h < 179:
            diff_h = diff_h + 1
        segment_hsv_point()

    if key == ord('s'):
        if diff_s > 5:
            diff_s = diff_s - 5
        segment_hsv_point()

    if key == ord('S'):
        if diff_s < 250:
            diff_s = diff_s + 5
        segment_hsv_point()

    if key == ord('v'):
        if diff_v > 5:
            diff_v = diff_v - 5
        segment_hsv_point()

    if key == ord('V'):
        if diff_v < 250:
            diff_v = diff_v + 5
        segment_hsv_point()


cv2.destroyAllWindows()
