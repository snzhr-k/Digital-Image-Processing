
import random
import cv2
import numpy as np

im = None
color_img = None
gray_img0 = None
gray_img = None
ffill_case = 1
lo_diff = 20
up_diff = 20
connectivity = 4
is_color = 1
is_mask = 0
new_mask_val = 255


def update_lo(pos):
    global lo_diff
    lo_diff = pos


def update_up(pos):
    global up_diff
    up_diff = pos


def on_mouse( event, x, y, flags, param ):
    global mask, color_img

    if color_img is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
            my_mask = None
            seed = (x, y)
            if ffill_case == 0:
                lo = up = 0
                flags = connectivity + (new_mask_val << 8)
            else:
                lo = lo_diff
                up = up_diff
                flags = connectivity + (new_mask_val << 8) + cv2.FLOODFILL_FIXED_RANGE

            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)

            if is_mask:
                my_mask = mask.copy()
                _, mask = cv2.threshold(my_mask, 1, 128, cv2.THRESH_BINARY)

            if is_color:
                color = (r, g, b)
                comp, color_img, mask, rect = cv2.floodFill(color_img, my_mask, seed, color, (lo, lo, lo), (up, up, up), flags)
                cv2.imshow('image', color_img)
            else:
                brightness = (r * 2 + g * 7 + b + 5) / 10
                comp, color_img, mask, rect = cv2.floodFill(gray_img, my_mask, seed, brightness, lo, up, flags)
                cv2.imshow('image', gray_img)

            cv2.imwrite('ffilldemo_color_img.png', color_img)
            print('{} pixels were repainted'.format(comp))

            if is_mask:
                cv2.imshow('mask', mask)
                cv2.imwrite('ffildemo_mask.png', mask)


if __name__ == "__main__":
    im = cv2.imread('fruits.jpg', cv2.IMREAD_COLOR)
    # im = cv2.imread('coins_rs.jpg', cv2.IMREAD_COLOR)
    print("Hot keys:")
    print("\tESC - quit the program")
    print("\tc - switch color/grayscale mode")
    print("\tm - switch mask mode")
    print("\tr - restore the original image")
    print("\ts - use null-range floodfill")
    print("\tf - use gradient floodfill with fixed(absolute) range")
    print("\tg - use gradient floodfill with floating(relative) range")
    print("\t4 - use 4-connectivity mode")
    print("\t8 - use 8-connectivity mode")

    color_img = im.copy()
    gray_img0 = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img0.copy()
    mask = np.ndarray((color_img.shape[0] + 2, color_img.shape[1] + 2), np.uint8)

    cv2.namedWindow('image', 1)
    cv2.createTrackbar('lo_diff', 'image', lo_diff, 255, update_lo)
    cv2.createTrackbar('up_diff', 'image', up_diff, 255, update_up)

    cv2.setMouseCallback('image', on_mouse)

    while True:
        if is_color:
            cv2.imshow('image', color_img)
        else:
            cv2.imshow('image', gray_img)

        c = cv2.waitKey(0) & 0xff
        if c == 27:
            print('Exiting ...')
            break
        elif c == ord('c'):
            if is_color:
                print('Grayscale mode is set')
                gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                is_color = 0
            else:
                print('Color mode is set')
                color_img = im.copy()
                mask[:, :] = 0
                is_color = 1

        elif c == ord('m'):
            if is_mask:
                cv2.destroyWindow('mask')
                is_mask = 0
            else:
                mask = np.ndarray((color_img.shape[0] + 2, color_img.shape[1] + 2), np.uint8)
                mask[:, :] = 0
                cv2.imshow('mask', mask)
                is_mask = 1

        elif c == ord('r'):
            print('Original image is restored')
            color_img = im.copy()
            mask = np.ndarray((color_img.shape[0] + 2, color_img.shape[1] + 2), np.uint8)
            mask[:, :] = 0
        elif c == ord('s'):
            print('Simple floodfill mode is set')
            ffill_case = 0
        elif c == ord('f'):
            print('Fixed Range floodfill mode is set')
            ffill_case = 1
        elif c == ord('g'):
            print('Gradient (floating range) floodfill mode is set')
            ffill_case = 2
        elif c == ord('4'):
            print('4-connectivity mode is set')
            connectivity = 4
        elif c == ord('8'):
            print('8-connectivity mode is set')
            connectivity = 8

    cv2.destroyAllWindows()
