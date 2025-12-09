
import cv2
import numpy as np


def create_gamma_lut(gamma):
    """Generate a 256-element lookup table corresponding to a Gamma parameter value.
    Because of the exponentiation, it must first be converted to [0, 1] and then back to [0, 255].
    """
    lut_out = np.arange(0, 256, 1, np.float32)
    lut_out = lut_out / 255.0
    lut_out = lut_out ** gamma
    lut_out = np.uint8(lut_out * 255.0)

    return lut_out


def apply_lut(im_in, lut_in, label_text):
    print(label_text)
    im_lut = cv2.LUT(im_in, lut_in, None)
    cv2.imshow('gamma', im_lut)
    cv2.waitKey(0)


im = cv2.imread('Sudoku_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_COLOR)
# im = cv2.imread('PalPant_800.jpg', cv2.IMREAD_COLOR)

# Gamma corrections

lut = create_gamma_lut(1.0 / 3.0)
apply_lut(im, lut, 'γ=1/3')

lut = create_gamma_lut(0.5)
apply_lut(im, lut, 'γ=1/2')

lut = np.arange(0, 256, 1, np.uint8)
apply_lut(im, lut, 'γ=1')

lut = create_gamma_lut(2.0)
apply_lut(im, lut, 'γ=2')

lut = create_gamma_lut(3.0)
apply_lut(im, lut, 'γ=3')

cv2.destroyAllWindows()
