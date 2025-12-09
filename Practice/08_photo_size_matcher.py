import cv2

file_im_in_name = 'webcam_selfie.jpg'
file_im_bg_name = 'street_dark'
file_im_bg_ext = '.jpg'
file_im_bg_save = file_im_bg_name + '_rs_cut.jpg'

im_in = cv2.imread(file_im_in_name, cv2.IMREAD_COLOR)
im_bg = cv2.imread(file_im_bg_name + file_im_bg_ext, cv2.IMREAD_COLOR)

in_h, in_w = im_in.shape[:2]
assert in_w > 0 and in_h > 0

bg_h, bg_w = im_bg.shape[:2]
assert bg_w > 0 and bg_h > 0

ratio_w = in_w / bg_w
ratio_h = in_h / bg_h

print(ratio_w, ratio_h)
ratio = max(ratio_w, ratio_h)

im_lanczos4 = cv2.resize(im_bg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LANCZOS4)
im_lanczos4_cut = im_lanczos4[0:in_h, 0:in_w]
print(im_lanczos4_cut.shape)

cv2.imshow('Original', im_in)
cv2.imshow('Resampled Lanczos', im_lanczos4)
cv2.imshow('Resampled Lanczos cut', im_lanczos4_cut)
cv2.imwrite(file_im_bg_save, im_lanczos4_cut)

cv2.waitKey(0)
cv2.destroyAllWindows()
