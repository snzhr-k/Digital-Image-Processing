
import numpy as np
import cv2

im_gray = cv2.imread('FrenchCardsShapes.png', cv2.IMREAD_GRAYSCALE)

val_thresh = 100

if val_thresh < 128:
    _, im_th = cv2.threshold(im_gray, val_thresh, 255, cv2.THRESH_BINARY_INV)
else:
    _, im_th = cv2.threshold(im_gray, val_thresh, 255, cv2.THRESH_BINARY)

moments, im_labels = cv2.connectedComponents(im_th, None, 8, cv2.CV_16U)

print('Number of components:', im_labels.max())

im_labels_norm = cv2.normalize(im_labels, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)
cv2.imshow('Labels', im_labels_norm)
cv2.waitKey(0)

im_comp = np.ndarray(im_gray.shape, np.uint8)
for idx in range(1, im_labels.max() + 1):
    print('=======================================================================')
    print('Component label:', idx)

    im_comp[:, :] = 0
    im_comp[im_labels == idx] = 255
    cv2.imshow('Component', im_comp)

    moments = cv2.moments(im_comp, True)
    print('Moments:', moments)
    print('Number of computed moments:', len(moments))

    hu_moments = cv2.HuMoments(moments).flatten()
    print('Hu moments:', hu_moments)
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
    print('Log transformed Hu moments', hu_moments_log)

    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
