import cv2

im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_COLOR)

# Image cropping; specify row and column range, cat head part
im_cropped = im[82:172, 396:486]
# Copy the cropped part to a new image of the same size
im[10:100, 20:110] = im_cropped

cv2.imshow('image', im)
cv2.imshow('cropped', im_cropped)

cv2.waitKey(0)
