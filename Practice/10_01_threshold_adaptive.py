
import cv2

im_src = cv2.imread('screen01_h.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Source', im_src)

im_thresh = cv2.adaptiveThreshold(im_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, -40)
print('Adaptive threshold with 35x35 block size and -40 constant value.')
cv2.imshow('Result', im_thresh)
cv2.waitKey(0)

cv2.destroyAllWindows()
