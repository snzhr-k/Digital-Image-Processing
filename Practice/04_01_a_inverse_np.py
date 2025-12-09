
import cv2

# im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_GRAYSCALE)
im = cv2.imread('Sudoku_rs.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', im)

# Inverse using arithmetics
im_inverse = 255 - im
cv2.imshow('Inverse', im_inverse)
cv2.waitKey(0)

cv2.destroyAllWindows()
