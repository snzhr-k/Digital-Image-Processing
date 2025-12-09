
import cv2

FILE_INPUT = 'DSC02620.jpg'
FILE_OUTPUT = 'DSC02620_rs.jpg'
MAX_WIDTH = 1000
MAX_HEIGHT = 800

im = cv2.imread(FILE_INPUT, cv2.IMREAD_COLOR)

h, w = im.shape[:2]
assert w > 0 and h > 0

print("Input image shape:", im.shape)

ratio_w = MAX_WIDTH / w
ratio_h = MAX_HEIGHT / h

if ratio_w * h > MAX_HEIGHT:
    ratio = ratio_h
else:
    ratio = ratio_w

im_lanczos4 = cv2.resize(im, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LANCZOS4)

print("New image shape:", im_lanczos4.shape)

cv2.imshow('Original', im)
cv2.imshow('Resampled Lanczos', im_lanczos4)
cv2.imwrite(FILE_OUTPUT, im_lanczos4)

cv2.waitKey(0)
cv2.destroyAllWindows()
