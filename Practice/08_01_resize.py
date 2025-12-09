
import cv2

par_fx = 3
par_fy = 1.5

im = cv2.imread('calb.png', cv2.IMREAD_COLOR)

im_nearest = cv2.resize(im, None, fx=par_fx, fy=par_fy, interpolation=cv2.INTER_NEAREST)
im_linear = cv2.resize(im, None, fx=par_fx, fy=par_fy, interpolation=cv2.INTER_LINEAR)
im_area = cv2.resize(im, None, fx=par_fx, fy=par_fy, interpolation=cv2.INTER_AREA)
im_cubic = cv2.resize(im, None, fx=par_fx, fy=par_fy, interpolation=cv2.INTER_CUBIC)
im_lanczos4 = cv2.resize(im, None, fx=par_fx, fy=par_fy, interpolation=cv2.INTER_LANCZOS4)

cv2.imshow('Original', im)
cv2.imshow('Resampled nearest', im_nearest)
cv2.imshow('Resampled linear', im_linear)
cv2.imshow('Resampled area', im_area)
cv2.imshow('Resampled cubic spline', im_cubic)
cv2.imshow('Resampled Lanczos', im_lanczos4)
cv2.waitKey(0)

cv2.destroyAllWindows()
