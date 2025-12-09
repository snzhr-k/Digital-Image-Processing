
import numpy as np
import cv2

np_arr = np.array([0, 50, 100, 125, 150, 200, 250], dtype=np.uint8)
print(np_arr)

_, res1 = cv2.threshold(np_arr, 125, 255, cv2.THRESH_BINARY)
print(res1.transpose())

_, res2 = cv2.threshold(np_arr, 125, 255, cv2.THRESH_BINARY_INV)
print(res2.transpose())

_, res3 = cv2.threshold(np_arr, 125, 255, cv2.THRESH_TRUNC)
print(res3.transpose())

_, res4 = cv2.threshold(np_arr, 125, 255, cv2.THRESH_TOZERO)
print(res4.transpose())

_, res5 = cv2.threshold(np_arr, 125, 255, cv2.THRESH_TOZERO_INV)
print(res5.transpose())
