import cv2
import numpy as np

path = "../Practice/car_numberplate_dirty_01.png"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', img)

kernel = np.ones((15, 15), np.uint8)

filtered = cv2.medianBlur(img, 15)


ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Thresh', thresh)

morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imshow('Morph', morph)

diff = cv2.absdiff(thresh, morph)
cv2.imshow('Diff', diff)

result = img.copy()
result[diff > 0] = filtered[diff > 0]
cv2.imshow('Result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()