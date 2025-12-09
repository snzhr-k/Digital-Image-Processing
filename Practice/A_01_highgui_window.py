
import cv2

im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_COLOR)

cv2.imshow('imshow', im)
cv2.waitKey(0)
cv2.destroyWindow('imshow')

cv2.namedWindow('namedWindow')
cv2.waitKey(0)
cv2.imshow('namedWindow', im)

cv2.setWindowTitle('namedWindow', 'namedWindow title')
cv2.moveWindow('namedWindow', 200, 300)

print('Select your Region of Interest! Left mouse button + move, then SPACE or ENTER. Cancel with c!')
cv2.setWindowTitle('namedWindow', 'Select ROI!')
roi = cv2.selectROI('namedWindow', im)
print(roi)

cv2.destroyAllWindows()
