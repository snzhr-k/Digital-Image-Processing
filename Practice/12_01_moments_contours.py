
import numpy as np
import cv2

im = cv2.imread('FrenchCardsShapes.png')
im_orig = im.copy()
im_gray = 255 - cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, im_thresh = cv2.threshold(im_gray, 127, 255, 0)

contours, hierarchy = cv2.findContours(im_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

print('Hierarchy:')
print(hierarchy)

print('Number of contours:', len(contours))
for cntr_idx in range(0, len(contours)):
    # Skip if not an outer contour (it has a parent)
    if hierarchy[0][cntr_idx][3] != -1:
        continue

    print('=======================================================================')
    print('Contour index:', cntr_idx)

    print(contours[cntr_idx].shape)
    moments = cv2.moments(contours[cntr_idx], True)
    print('Moments:', moments)
    print('Number of computed moments:', len(moments))
    print('m00:', moments['m00'])

    hu_moments = cv2.HuMoments(moments).flatten()
    print('Hu moments:', hu_moments)
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
    print('Log transformed Hu moments', hu_moments_log)

    cv2.drawContours(im, contours, cntr_idx, (0, 255, 0), 3)
    cv2.imshow('Contours', im)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

    im = im_orig.copy()

cv2.destroyAllWindows()
