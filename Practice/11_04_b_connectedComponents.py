import cv2

im_gray = cv2.imread('FrenchCardsShapes.png', cv2.IMREAD_GRAYSCALE)

thresh_val = 100

if thresh_val < 128:
    _, im_th = cv2.threshold(im_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
else:
    _, im_th = cv2.threshold(im_gray, thresh_val, 255, cv2.THRESH_BINARY)

num_comp, im_labels, stats, centroid = cv2.connectedComponentsWithStats(im_th, None, 8, cv2.CV_16U)

print('Number of components:', num_comp)
print(stats)
print('Area of component with index [1]:', stats[1][cv2.CC_STAT_AREA])
print(centroid)

im_labels_norm = cv2.normalize(im_labels, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)
cv2.imshow('Labels', im_labels_norm)
cv2.waitKey(0)

cv2.destroyAllWindows()
