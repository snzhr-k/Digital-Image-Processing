
import numpy as np
import cv2

# https://www.pyimagesearch.com/2014/10/27/opencv-shape-descriptor-hu-moments-example/
# https://docs.opencv.org/4.10.0/d3/dc0/group__imgproc__shape.html#gaf2b97a230b51856d09a2d934b78c015f

component_selected = False
component_selection_string = 'First, select a reference shape for comparison by right-clicking on it!'
component_compare_string = 'Left mouse button: select components for comparison.\n' \
                           'Right mouse button: select a new reference component.'


def initialize_component_shape(x, y):
    print('initialize_component_shape() called!')
    global im_labels, im_pattern_component

    print('=======================================================================')
    idx = im_labels[y, x]
    print('Selected component label:', idx)
    if idx == 0:
        print('Background, not good!')
        return

    im_pattern_component[:, :] = 0
    im_pattern_component[im_labels == idx] = 255
    cv2.imshow('Pattern', im_pattern_component)

    moments = cv2.moments(im_pattern_component, True)
    # print('Moments:', moments)
    # print('Number of computed moments:', len(moments))

    hu_moments = cv2.HuMoments(moments).flatten()
    print('Hu moments:', hu_moments)
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
    print('Log transformed Hu moments', hu_moments_log)


def mouse_click(event, x, y, flags, param):
    # Taking global variables
    global im_th, im_labels, stats, centroids
    global im_pattern_component
    global component_selected, component_selection_string, component_compare_string

    if event == cv2.EVENT_RBUTTONUP:
        initialize_component_shape(x, y)
        component_selected = True
        print(component_compare_string)
        return

    if event == cv2.EVENT_LBUTTONUP:
        if not component_selected:
            print(component_selection_string)
            return

        print('=======================================================================')
        idx = im_labels[y, x]
        print('Component label:', idx)
        if idx == 0:
            print('Background!')
            return

        print(stats[idx])
        print('Enclosing rectangle top left corner: ({}, {})'.format(stats[idx][cv2.CC_STAT_LEFT],
                                                                    stats[idx][cv2.CC_STAT_TOP]))
        print('Enclosing rectangle size: {}x{}'.format(stats[idx][cv2.CC_STAT_WIDTH],
                                                        stats[idx][cv2.CC_STAT_HEIGHT]))
        print('Component area in pixels: {}'.format(stats[idx][cv2.CC_STAT_AREA]))
        print('Component center of gravity:', centroids[idx])

        im_comp[:, :] = 0
        im_comp[im_labels == idx] = 255
        cv2.imshow('Component', im_comp)
        cv2.imshow('Pattern', im_pattern_component)

        moments = cv2.moments(im_comp, True)
        # print('Moments:', moments)
        # print('Number of computed moments:', len(moments))

        hu_moments = cv2.HuMoments(moments).flatten()
        print('Hu moments:', hu_moments)
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        print('Log transformed Hu moments:', hu_moments_log)

        shape_match_value = cv2.matchShapes(im_pattern_component, im_comp, cv2.CONTOURS_MATCH_I2, 0)
        print('Shape match:', shape_match_value * 1000)

        print(component_compare_string)


im_gray = cv2.imread('shapes_crowns_arrows.png', cv2.IMREAD_GRAYSCALE)
# im_gray = cv2.imread('FrenchCardsShapes.png', cv2.IMREAD_GRAYSCALE)
# im_gray = cv2.imread('FCards_01_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im_gray = cv2.imread('FCards_02_rs.jpg', cv2.IMREAD_GRAYSCALE)
im_pattern_component = np.ndarray(im_gray.shape, np.uint8)

val_thresh = 150  # shapes_crowns_arrows.png esetén
# val_thresh = 100  # FrenchCardShapes.png, FCards_01_rs.jpg, és FCards_02_rs.jpg esetén

if val_thresh < 128:
    _, im_th = cv2.threshold(im_gray, val_thresh, 255, cv2.THRESH_BINARY_INV)
else:
    _, im_th = cv2.threshold(im_gray, val_thresh, 255, cv2.THRESH_BINARY)

num_comp, im_labels, stats, centroids = cv2.connectedComponentsWithStats(im_th, None, 8, cv2.CV_16U)

print('Number of components:', num_comp)
print(component_selection_string)

im_labels_norm = cv2.normalize(im_labels, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)
_, im_labels_norm_thresh = cv2.threshold(im_labels_norm, 0, 65535, cv2.THRESH_BINARY)
cv2.imshow('Labels', im_labels_norm_thresh)
im_comp = np.ndarray(im_gray.shape, np.uint8)
cv2.setMouseCallback('Labels', mouse_click)
cv2.waitKey(0)

cv2.destroyAllWindows()
