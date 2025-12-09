
import cv2


def on_tb_changed_w(pos):
    w = pos / 10
    print(w)
    im_sharpen = cv2.add(im, w * im_diff, dtype=cv2.CV_8UC1)
    cv2.imshow('sharpen', im_sharpen)


# im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_COLOR)
im = cv2.imread('Hermes_h.jpg', cv2.IMREAD_COLOR)
# im = cv2.imread('webcam_selfie.jpg', cv2.IMREAD_COLOR)

im_blur = cv2.GaussianBlur(im, (5, 5), 2.0)
im_diff = cv2.subtract(im, im_blur, dtype=cv2.CV_16S)

cv2.imshow('im', im)
cv2.imshow('im_blur', im_blur)
# We display the original so that the window size is appropriate for the trackbar.
cv2.imshow('sharpen', im)
cv2.createTrackbar('W', 'sharpen', 25, 50, on_tb_changed_w)

cv2.waitKey(0)
cv2.destroyAllWindows()
