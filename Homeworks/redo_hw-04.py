import cv2
import numpy as np

path = 'ImProcExamples2025/hk_flower_h.jpg'
im = cv2.imread(path)

canny = cv2.Canny(im, 100, 200)
cv2.imshow('Canny Edges', canny)

blurred = cv2.GaussianBlur(im, (5, 5), 0)
canny_blurred = cv2.Canny(blurred, 100, 200)
canny_blurred_bgr = cv2.cvtColor(canny_blurred, cv2.COLOR_RGB2BGR)

print(canny_blurred.shape)
print(canny_blurred_bgr.shape)
print(blurred.shape)


mask = cv2.bitwise_not(canny_blurred)
cv2.imshow('Mask', mask)

cartoon = cv2.bitwise_and(blurred, blurred, mask=mask)
cv2.imshow('Cartoon', cartoon)

cv2.waitKey(0)
cv2.destroyAllWindows()