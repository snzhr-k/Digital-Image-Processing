import cv2
import numpy as np

im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_UNCHANGED)
rows, cols = im.shape[:2]

mtx_trf = np.float32([[1, 0, 100],
                      [0, 1, 50]])
im_trf = cv2.warpAffine(im, mtx_trf, (cols, rows))

cv2.imshow('im_trf', im_trf)
cv2.waitKey(0)

cv2.destroyAllWindows()
