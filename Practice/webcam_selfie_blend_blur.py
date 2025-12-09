import cv2
import numpy as np

im_bgr = cv2.imread('webcam_selfie/webcam_selfie.jpg', cv2.IMREAD_COLOR)
im_mask = cv2.imread('webcam_selfie/webcam_selfie_mask.png', cv2.IMREAD_COLOR)
im_bg = cv2.imread('webcam_selfie/webcam_selfie_bg.jpg', cv2.IMREAD_COLOR)

assert im_bgr.shape[:2] == im_mask.shape[:2]
assert im_bgr.shape[:2] == im_bg.shape[:2]

im_mask_blur = cv2.GaussianBlur(im_mask, (5, 5), 2.0)

cv2.imshow('im_bgr', im_bgr)
cv2.imshow('im_mask_blur', im_mask_blur)
cv2.imshow('im_bg', im_bg)

im_weight1 = np.float32(im_mask_blur) / 255.0
im_weight2 = 1.0 - im_weight1
im_res = np.uint8(im_weight1 * im_bgr + im_weight2 * im_bg)

cv2.imshow('im_res', im_res)
cv2.imwrite('webcam_selfie_blend_blur.jpg', im_res)

cv2.waitKey(0)
cv2.destroyAllWindows()
