
# OpenCV2 image reading, grayscale conversion
# OpenCV online documentation: https://docs.opencv.org/

# Importing OpenCV package definitions
import cv2

# Reading image from file as grayscale
im_gray = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)

# Printing matrix size to console
print('im_gray.shape:', im_gray.shape)

# Displaying image in a window
cv2.imshow('im_gray', im_gray)
cv2.waitKey(0)

# Reading image from file as color
im_color = cv2.imread('OpenCV-logo.png', cv2.IMREAD_COLOR)
print('im_color.shape:', im_color.shape)
im_gray2 = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
print('im_gray2.shape:', im_gray2.shape)

# Displaying image in a window
cv2.imshow('im_gray2', im_gray2)
cv2.waitKey(0)

cv2.destroyAllWindows()
