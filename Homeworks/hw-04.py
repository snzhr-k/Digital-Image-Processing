import cv2
import numpy as np
src = ("../Practice/hk_flower_h.jpg")
img = cv2.imread(src)


kernel = np.ones((7,7), np.uint8)
smooth = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

edges = cv2.Canny(smooth, 80, 150)

kernel = np.ones((3,3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)

edges_bgr = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR)
mask = cv2.bitwise_not(edges_bgr)
cartoon = cv2.bitwise_and(smooth, mask)

cv2.imshow("Original", img)
cv2.imshow("Smoothed", smooth)
cv2.imshow("Edges", edges)
cv2.imshow("Dilated edges", edges_dilated)
cv2.imshow("Final Cartoon", cartoon)

cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.waitKey(0)
cv2.destroyAllWindows()
