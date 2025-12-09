
# Visualize the Hough transformation
# Attila Tanács, 2020-2025
# University of Szeged, Hungary

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

point_list = []
plt_color_list = ['y', 'c', 'm', 'r', 'g', 'b', 'k']
# pts_color_list = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]
pts_color_list = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
hough_param_clicked = []


def cross(p1, p2):
    if len(p1) != 2 and len(p2) !=2:
        print('Size problem!')

    return p1[0] * p2[1] - p1[1] * p2[0]


def get_diagram_as_image(fig_in):
    fig_in.canvas.draw()
    data = np.array(fig_in.canvas.renderer.buffer_rgba())
    data_bgr = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)

    return data_bgr


def visualize_rho_theta():
    global P1, rho, theta, im
    global fig, ax
    global point_list
    global hough_acc, hough_param_clicked

    im_cp = im.copy()
    # color_idx = len(point_list) - 1

    a = math.cos(math.radians(theta))
    b = math.sin(math.radians(theta))
    P2 = np.asarray(P1) + np.asarray((b, -a))
    p1 = np.asarray(P1)
    p2 = np.asarray(P2)
    p3 = np.asarray((0, 0))
    # Computing signed distance
    rho = (cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    print('r =', rho)

    x0 = a * rho
    y0 = b * rho
    # Computing line endpoints outside of image matrix
    pt1 = (0, 0)
    pt2 = (int(x0), int(y0))
    cv2.line(im_cp, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
    pt1 = (int(x0 + size_max * (-b)), int(y0 + size_max * a))
    pt2 = (int(x0 - size_max * (-b)), int(y0 - size_max * a))
    cv2.line(im_cp, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    for idx, pnt in enumerate(point_list):
        cv2.circle(im_cp, pnt, 5, pts_color_list[idx % 7], -1)

    hough_acc_work = hough_acc.copy()
    if len(hough_param_clicked) > 0:
        t = hough_param_clicked[0][0]
        r = hough_param_clicked[0][1]

        h_y = int((r + size_max) * 0.125)
        h_x = t >> 1
        hough_acc_work[h_y, :] = 255
        hough_acc_work[:, h_x] = 255

        a = math.cos(math.radians(t))
        b = math.sin(math.radians(t))
        x0 = a * r
        y0 = b * r
        print('Green line parameters:', r, t)
        # Computing line endpoints outside of image matrix
        pt1 = (int(x0 + size_max * (-b)), int(y0 + size_max * a))
        pt2 = (int(x0 - size_max * (-b)), int(y0 - size_max * a))
        cv2.line(im_cp, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('im', im_cp)

    im_diag = get_diagram_as_image(fig)
    cv2.imshow('Hough parameter space (theta-rho)', im_diag)

    hough_acc_rs = cv2.resize(hough_acc_work, None, None, 4, 4, cv2.INTER_NEAREST)
    cv2.imshow('Hough accumulator', hough_acc_rs)


def on_theta_trackbar_change(x):
    global rho, theta
    theta = x
    a = math.cos(math.radians(theta))
    b = math.sin(math.radians(theta))
    P2 = np.asarray(P1) + np.asarray((b, -a))
    p1 = np.asarray(P1)
    p2 = np.asarray(P2)
    p3 = np.asarray((0, 0))
    # Computing signed distance
    rho = (cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    hough_acc[int((rho + size_max) * 0.125), theta >> 1] += 32
    color_idx = len(point_list) - 1
    ax.scatter(theta, rho, s=1, color=plt_color_list[color_idx % 7])
    visualize_rho_theta()


def on_mouse_event(event, x, y, param, flag):
    global P1, point_list, rho, theta

    if event == cv2.EVENT_LBUTTONDOWN:
        P1 = (x, y)
        print('Point added:', P1)
        point_list.append(P1)
        visualize_rho_theta()


def on_hough_mouse_event(event, x, y, param, flag):
    global hough_param_clicked

    if event == cv2.EVENT_MOUSEMOVE:
        hpt_theta = 2 * int(x / 4)     # clicked theta
        hpt_rho = int(4 * int(y / 2) - int(size_max))
        hpt = (hpt_theta, hpt_rho)
        hough_param_clicked = [hpt]
        visualize_rho_theta()


im = cv2.imread('sudoku_rs.jpg', cv2.IMREAD_COLOR)
edge = cv2.Canny(im, 50, 300)
im = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
size_max = math.sqrt(im.shape[0] ** 2 + im.shape[1] ** 2)
# P1 = (img.shape[1] >> 1, img.shape[0] >> 1)
P1 = (168, 100)
point_list.append(P1)
theta = 0

fig = plt.figure(figsize=(6, 6), dpi=80)
fig.canvas.manager.set_window_title('Hough parameter space')
ax = fig.add_subplot(111)
plt.xlim([0, 180])
plt.ylim([-size_max, size_max])
plt.xlabel('Theta')
plt.ylabel('Rho')

rho = 0
hough_acc = np.zeros((int(0.25 * size_max + 0.5), 90), np.uint8)
cv2.imshow('Hough accumulator', hough_acc)
cv2.namedWindow('im')
cv2.createTrackbar('theta', 'im', theta, 179, on_theta_trackbar_change)
cv2.imshow('im', im)
on_theta_trackbar_change(theta)
cv2.setMouseCallback('im', on_mouse_event)
cv2.setMouseCallback('Hough accumulator', on_hough_mouse_event)

while True:
    key = cv2.waitKey(0)

    if key == ord('q'):
        break

    if key == ord('a'):
        for val in range(0, 180, 2):
            hough_param_clicked = []
            cv2.setTrackbarPos('theta', 'im', val)
            cv2.waitKey(1)

    if key == ord('p'):
        add_points = [(226, 52), (47, 86), (272, 110), (211, 191), (221, 104), (69, 179)]
        for pt in add_points:
            P1 = pt
            print('Point added:', pt)
            point_list.append(pt)
            for val in range(0, 180, 2):
                hough_param_clicked = []
                cv2.setTrackbarPos('theta', 'im', val)
                cv2.waitKey(1)


cv2.destroyAllWindows()
