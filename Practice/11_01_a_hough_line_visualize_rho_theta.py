
import cv2
import math


def visualize_rho_theta():
    global rho, theta, im

    im_cp = im.copy()
    a = math.cos(math.radians(theta))
    b = math.sin(math.radians(theta))
    x0 = a * rho
    y0 = b * rho
    # Computing line endpoints outside of image matrix
    pt1 = (0, 0)
    pt2 = (int(x0), int(y0))
    cv2.line(im_cp, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
    pt1 = (int(x0 + size_max * (-b)), int(y0 + size_max * a))
    pt2 = (int(x0 - size_max * (-b)), int(y0 - size_max * a))
    cv2.line(im_cp, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('im', im_cp)


def on_rho_trackbar_change(x):
    global rho

    rho = x
    visualize_rho_theta()


def on_theta_trackbar_change(x):
    global theta

    theta = x
    visualize_rho_theta()


im = cv2.imread('sudoku_rs.jpg', cv2.IMREAD_COLOR)
size_max = math.sqrt(im.shape[0] ** 2 + im.shape[1] ** 2)
rho = im.shape[0] >> 1
theta = 45

cv2.imshow('im', im)

cv2.createTrackbar('r (rho)', 'im', rho, int(size_max), on_rho_trackbar_change)
cv2.createTrackbar('theta', 'im', theta, 180, on_theta_trackbar_change)

cv2.waitKey(0)
cv2.destroyAllWindows()
