
import cv2

im = cv2.imread('calb.png', cv2.IMREAD_COLOR)
print(im.shape)

dsize = (200, 100)
im_resize = cv2.resize(im, dsize, interpolation=cv2.INTER_AREA)

fx = dsize[0] / im.shape[1]
fy = dsize[1] / im.shape[0]

print('fx =', fx)
print('fy =', fy)

cv2.imshow('Original', im)
cv2.imshow('Resampled area', im_resize)
cv2.waitKey(0)

cv2.destroyAllWindows()
