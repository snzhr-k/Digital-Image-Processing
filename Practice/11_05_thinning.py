
import cv2

im = cv2.imread('binary_blobs.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', im)

im_thin = cv2.ximgproc.thinning(im, None, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
# im_thin = cv2.ximgproc.thinning(im, None, thinningType=cv2.ximgproc.THINNING_GUOHALL)
cv2.imshow('thinning', im_thin)

im_bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
im_b, im_g, im_r = cv2.split(im_bgr)
im_b = cv2.bitwise_and(im_b, ~im_thin)
im_g = cv2.bitwise_and(im_g, ~im_thin)
im_bgr = cv2.merge((im_b, im_g, im_r))
cv2.imshow('Overlay', im_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()
