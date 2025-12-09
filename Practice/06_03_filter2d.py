import cv2
import numpy as np

im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)

noise = np.zeros(im.shape, np.int16)
cv2.randn(noise, 0.0, 20.0)

im_noise = cv2.add(im, noise, dtype=cv2.CV_8UC1)
cv2.imshow('noise', im_noise)

# kernel = np.array([[-1, -1, -1],
#                    [-1, 9, -1],
#                    [-1, -1, -1]])

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

kernel = kernel / 9.0

print(kernel)

im_noise_filter = cv2.filter2D(im_noise, -1, kernel)
cv2.imshow('filter2d', im_noise_filter)
cv2.waitKey(0)

cv2.destroyAllWindows()
