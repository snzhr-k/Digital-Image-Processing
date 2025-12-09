
import cv2
import numpy as np
from matplotlib import pyplot as plt

plot_type = 2


def print_usage():
    print('Usage:')
    print('x: Select intensity profile along the X-axis')
    print('y: Select intensity profile along the Y-axis')
    print('b: Select intensity profile along both axes (default)')
    print('Left mouse click: draw selected profile(s) defined by the clicked position')
    print('q: quit')


def get_diagram_as_image(fig_in):
    fig_in.canvas.draw()
    data = np.array(fig_in.canvas.renderer.buffer_rgba())
    data_bgr = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)

    return data_bgr


def mouse_click(event, x, y, flags, param):
    global im_in, im_gray_bgr, im_copy
    global fig

    if event == cv2.EVENT_LBUTTONDOWN:
        im_gray_bgr = im_copy.copy()
        if im_gray_bgr.ndim == 3:
            if (plot_type == 0) or (plot_type == 2):
                im_gray_bgr[y, :] = [0, 0, 255]
            if (plot_type == 1) or (plot_type == 2):
                im_gray_bgr[:, x] = [0, 255, 0]
        cv2.imshow('im', im_gray_bgr)

        fig.clf()
        if (plot_type == 0) or (plot_type == 2):
            plt.plot(im_in[y, :], color='r')
        if (plot_type == 1) or (plot_type == 2):
            plt.plot(im_in[:, x], color='g')
        # plt.ylim([0, 255])
        # fig.tight_layout(pad=0)
        edge_plot = get_diagram_as_image(fig)
        cv2.imshow('edge_plot', edge_plot)


print_usage()

im_in = cv2.imread('shapes/shapes_10.jpg', cv2.IMREAD_GRAYSCALE)
# im_in = cv2.imread('tree_binary.png', cv2.IMREAD_GRAYSCALE)
# im_in = cv2.imread('tree_blur_02.png', cv2.IMREAD_GRAYSCALE)
# im_in = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# im_in = cv2.imread('PalPant_800.jpg', cv2.IMREAD_GRAYSCALE)
# im_in = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
im_gray_bgr = cv2.merge([im_in, im_in, im_in])
im_copy = im_gray_bgr.copy()

fig = plt.figure(figsize=(4, 2), dpi=100)

cv2.imshow('im', im_in)
cv2.setMouseCallback('im', mouse_click)

while True:
    key = cv2.waitKey(0)

    if key == ord('x'):
        plot_type = 0

    if key == ord('y'):
        plot_type = 1

    if key == ord('b'):
        plot_type = 2

    if key == ord('q'):
        break


cv2.destroyAllWindows()
