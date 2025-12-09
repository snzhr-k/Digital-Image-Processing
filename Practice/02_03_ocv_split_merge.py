
# OpenCV2 reading image, color space conversion
# OpenCV online documentation: https://docs.opencv.org/4.12.0/de/d25/imgproc_color_conversions.html

# Importing OpenCV package definitions
import cv2

# Reading image from file
im_bgr = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_COLOR)
cv2.imshow('im_bgr', im_bgr)

# Grayscale
im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
cv2.imshow('im_gray', im_gray)
cv2.waitKey(0)

# Splitting into color channels and display
im_b, im_g, im_r = cv2.split(im_bgr)
cv2.imshow('red', im_r)
cv2.imshow('green', im_g)
cv2.imshow('blue', im_b)
cv2.waitKey(0)

# Red channel zeroing and BGR image generation
im_r[:, :] = 0
im_bgr2 = cv2.merge((im_b, im_g, im_r))
cv2.imshow('bg0', im_bgr2)
cv2.imwrite('Golyo_GB.jpg', im_bgr2)
cv2.waitKey(0)

# Close color channel windows
cv2.destroyWindow('red')
cv2.destroyWindow('green')
cv2.destroyWindow('blue')
cv2.destroyWindow('bg0')

# Converting to the Lab color space
# L: grayscale intensity
# a, b: chromaticity (color information)
im_Lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2Lab)
print('im_Lab.shape:', im_Lab.shape)
cv2.imshow('im_Lab', im_Lab)

    # Splitting Lab channels
im_L, im_a, im_b = cv2.split(im_Lab)
cv2.imshow('im_L', im_L)
cv2.imshow('im_a', im_a)
cv2.imshow('im_b', im_b)
cv2.waitKey(0)

# Zeroing im_b and assemble new BGR image
im_b.fill(0)
im_Lab2 = cv2.merge([im_L, im_a, im_b])
im_bgr2 = cv2.cvtColor(im_Lab2, cv2.COLOR_Lab2BGR)
cv2.imshow('im_bgr2', im_bgr2)
cv2.waitKey(0)

cv2.destroyAllWindows()
