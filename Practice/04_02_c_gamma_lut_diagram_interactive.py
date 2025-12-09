
# https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

import cv2
import numpy as np
from matplotlib import pyplot as plt


def on_trackbar(pos):
    global im

    gamma = (pos / 10.0) - 3
    if gamma < 1:
        gamma = -1.0 / (gamma - 2.0)

    lut = create_gamma_lut(gamma)
    apply_lut(im, lut, 'γ={0:2.2f}'.format(gamma))


def get_diagram_as_image(fig_in):
    fig_in.canvas.draw()
    data = np.array(fig_in.canvas.renderer.buffer_rgba())
    data_bgr = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)

    return data_bgr


def create_gamma_lut(gamma):
    """Generate a 256-element lookup table corresponding to a Gamma parameter value.
    Because of the exponentiation, it must first be converted to [0, 1] and then back to [0, 255].
    """
    lut_out = np.arange(0, 256, 1, np.float32)
    lut_out = lut_out / 255.0
    lut_out = lut_out ** gamma
    lut_out = np.uint8(lut_out * 255.0)

    return lut_out


def apply_lut(image, lut, label_text):
    global x, fig

    print(label_text)

    im_lut = cv2.LUT(image, lut, None)
    cv2.imshow('Gamma', im_lut)

    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_title('Gamma correction LUT diagram')
    plt.plot(x, x, 'g--', label='Original')
    plt.plot(x, lut, 'r-', label=label_text)
    plt.xlabel('Original intensity values')
    plt.ylabel('Gamma corrected value')
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.legend()

    lut_diag = get_diagram_as_image(fig)
    cv2.imshow('Gamma LUT', lut_diag)


# im = cv2.imread('Sudoku_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_COLOR)
im = cv2.imread('PalPant_800.jpg', cv2.IMREAD_COLOR)

# Base points [0, 255]
x = np.arange(0, 256, 1, np.uint8)

# Diagram settings
fig = plt.figure(figsize=(6, 6), dpi=80)
fig.canvas.manager.set_window_title('Gamma correction LUT diagram')

# Gamma correction

cv2.imshow('Gamma', im)
cv2.createTrackbar('Parameter', 'Gamma', 40, 80, on_trackbar)
cv2.waitKey(0)

cv2.destroyAllWindows()
