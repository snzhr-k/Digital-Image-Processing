import cv2
import numpy as np

file_in_base_name = 'webcam_selfie/webcam_selfie'
file_in_name_ext = '.jpg'
file_mask_base_name = file_in_base_name + '_mask'
file_mask_name_ext = '.png'
mouse_left_down = False
circle_radius = 10
circle_color = 255


def draw_and_show(x, y):
    global im, im_orig, im_mask
    global mouse_left_down
    global circle_radius
    global last_x, last_y

    if last_x >= 0 and last_y >= 0:
        cv2.circle(im_mask, (x, y), circle_radius, [circle_color], -1)

        im = im_orig.copy()
        im[im_mask == 255, 2] = 255

    cv2.imshow('im_mask', im_mask)
    cv2.imshow('im', im)

    last_x = x
    last_y = y


def fill_mask_holes(im_mask_in):
    mh, mw = im_mask_in.shape[:2]

    # Background border around mask image
    im_mask_temp = np.zeros((mh + 2, mw + 2), np.uint8)
    im_mask_temp[1:mh + 1, 1:mw + 1] = im_mask_in

    # Flood fill and combining with original mask
    im_mask_ff = np.zeros((mh + 4, mw + 4), np.uint8)
    cv2.floodFill(im_mask_temp, im_mask_ff, (0, 0), [255])
    im_mask_ff_inv = cv2.bitwise_not(im_mask_ff * 255)
    im_mask_filled = im_mask_in | im_mask_ff_inv[2:mh + 2, 2: mw + 2]

    return im_mask_filled


def mouse_handler(event, x, y, flags, param):
    global im_mask, im_mask_prev
    global mouse_left_down

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_left_down = True
        draw_and_show(x, y)
        im_mask_prev = im_mask.copy()

    if event == cv2.EVENT_LBUTTONUP:
        mouse_left_down = False

    if event == cv2.EVENT_MOUSEMOVE and mouse_left_down:
        draw_and_show(x, y)
        im_mask_prev = im_mask.copy()

    if event == cv2.EVENT_MOUSEMOVE and not mouse_left_down:
        im_mask = im_mask_prev.copy()
        draw_and_show(x, y)


im_bgr = cv2.imread(file_in_base_name + file_in_name_ext, cv2.IMREAD_COLOR)

if im_bgr is None:
    print('Cannot read image file', file_in_base_name + file_in_name_ext)
    exit(-1)

im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
im_gray_bgr = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)
h, w = im_bgr.shape[:2]
im = im_gray_bgr.copy()
im_orig = im.copy()

cv2.imshow('im', im)
cv2.setMouseCallback('im', mouse_handler)

im_mask = np.ndarray((h, w), np.uint8)
im_mask.fill(0)
cv2.imshow('im_mask', im_mask)
im_mask_prev = im_mask.copy()

last_x = -1000
last_y = -1000

print('Usage:')
print('C: change to color display')
print('G: change to grayscale display')
print('B: mask color to background (black)')
print('F: mask color to foreground (white)')
print('E: Fill mask with current mask color (erase)')
print('H: Fill mask foreground holes')
print('S: Save mask image')
print('L: reload input image')
print('+: increase mask circle size')
print('-: decrease mask circle size')
print('Q or ESC: quit')

while True:
    key = cv2.waitKey(0)

    if key == ord('c'):
        print('Change to color display.')
        im_orig = im_bgr.copy()
        draw_and_show(last_x, last_y)

    if key == ord('g'):
        print('Change to grayscale display.')
        im_orig = im_gray_bgr.copy()
        draw_and_show(last_x, last_y)

    if key == ord('b'):
        circle_color = 0
        draw_and_show(last_x, last_y)
        print('Mask color changed to background (black).')

    if key == ord('f'):
        circle_color = 255
        draw_and_show(last_x, last_y)
        print('Mask color changed to foreground (white).')

    if key == ord('e'):
        print('Erasing mask.')
        im_mask.fill(circle_color)
        im_mask_prev = im_mask.copy()
        draw_and_show(last_x, last_y)

    if key == ord('h'):
        print('Filling holes.')
        im_mask = fill_mask_holes(im_mask_prev)
        im_mask_prev = im_mask.copy()
        draw_and_show(last_x, last_y)

    if key == ord('s'):
        print('Writing mask to file.')
        cv2.imwrite(file_mask_base_name + file_mask_name_ext, im_mask_prev)

    if key == ord('l'):
        print('Loading image from file.')
        im_mask = cv2.imread(file_mask_base_name + file_mask_name_ext, cv2.IMREAD_GRAYSCALE)
        if im_mask is None:
            im_mask = np.ndarray((h, w), np.uint8)
            im_mask.fill(0)

        im_mask_prev = im_mask.copy()
        draw_and_show(last_x, last_y)

    if key == ord('+'):
        circle_radius += 5
        if circle_radius > 100:
            circle_radius = 100

        im_mask = im_mask_prev.copy()
        draw_and_show(last_x, last_y)
        print('circle_radius:', circle_radius)

    if key == ord('-') or key == 95:
        circle_radius -= 5
        if circle_radius < 5:
            circle_radius = 5

        im_mask = im_mask_prev.copy()
        draw_and_show(last_x, last_y)
        print('circle_radius:', circle_radius)

    if key == 27 or key == ord('q'):
        print('Exiting.')
        break

cv2.destroyAllWindows()
