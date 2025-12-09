
# OpenCV2: image reading, displaying, mirroring
# OpenCV online documentation: https://docs.opencv.org/

# Importing OpenCV package definitions
import cv2

# Printing OpenCV version number
print('OpenCV version:', cv2.__version__)

# Reading image from file
im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_COLOR)

# Printing image size
print(im.shape)

# Displaying image in a window
cv2.imshow('image', im)
cv2.waitKey(0)

# Mirroring to the vertical middle axis, dispalying the result, writing to file
im_flipped = cv2.flip(im, 1)
cv2.imshow('image', im_flipped)
cv2.imwrite('OpenCV-logo-flipped.png', im_flipped)
cv2.waitKey(0)

# Mirroring to both middle axes
im_flipped2 = cv2.flip(im, -1)
cv2.imshow('image', im_flipped2)
cv2.waitKey(0)

# Closing all windows
cv2.destroyAllWindows()
