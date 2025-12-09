
import cv2

im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('tree_blur_02.png', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)

im_blur = cv2.GaussianBlur(im, (5, 5), 2.0)
im_edges = cv2.Canny(im_blur, 100, 200, None, 5, True)
cv2.imshow('Canny', im_edges)
cv2.waitKey(0)

cv2.destroyAllWindows()
