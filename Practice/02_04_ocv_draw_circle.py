
# Create, display and save OpenCV2 image matrix
# OpenCV online documentation: https://docs.opencv.org/

# Importing package definitions
import numpy as np
import cv2

# Create 320x200x3 Numpy array for BGR color image
im = np.ndarray((200, 320, 3), np.uint8)
# Fill in matrix with value 192 (light grey)
im.fill(192)
# Drawing a circle with (50, 100) center, 40 radius, red color, filled
cv2.circle(im, (50, 100), 40, (0, 0, 192), -1)
# Further drawing functions:
#   https://docs.opencv.org/4.12.0/dc/da5/tutorial_py_drawing_functions.html

# Show image in window
cv2.imshow('im', im)
cv2.waitKey(0)

# Save image to file
cv2.imwrite('ocv_test1_out.png', im)

# Close all windows
cv2.destroyAllWindows()
