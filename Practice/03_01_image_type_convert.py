
import cv2
import numpy as np

# Read and display image
im_uint8 = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
print('im_uint8[100, 100] =', im_uint8[100, 100])
cv2.imwrite('logo-uint8.png', im_uint8)
cv2.imshow('im_uint8', im_uint8)
cv2.waitKey(0)

# Convert to to uint16
im_uint16 = np.uint16(im_uint8)
print('im_uint16[100, 100] =', im_uint16[100, 100])
cv2.imshow('im_uint16', im_uint16)
cv2.imwrite('logo-uint16.png', im_uint16)
cv2.waitKey(0)

# uint16 multiply by 256
im_uint16_mult256 = im_uint16 * 256
print('im_uint16[100, 100] =', im_uint16_mult256[100, 100])
cv2.imshow('im_uint16_mult256', im_uint16_mult256)
cv2.imwrite('logo-uint16_mult256.png', im_uint16_mult256)
cv2.waitKey(0)

# Covert to int16
im_int16 = np.int16(im_uint8)
print('im_int16[100, 100] =', im_int16[100, 100])
cv2.imshow('im_int16', im_int16)
cv2.imwrite('logo-int16.png', im_int16)
cv2.waitKey(0)

# Convert to float32
im_float32 = np.float32(im_uint8)
print('im_float32[100, 100] =', im_float32[100, 100])
cv2.imshow('im_float32', im_float32)
cv2.imwrite('logo-float32.png', im_float32)
cv2.waitKey(0)

im_float32f_norm = cv2.normalize(im_float32, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
print('im_float32_norm[100, 100] =', im_float32f_norm[100, 100])
cv2.imshow('im_float32_norm', im_float32f_norm)
cv2.imwrite('logo-float32_norm.png', im_float32f_norm)
cv2.waitKey(0)

cv2.destroyAllWindows()
