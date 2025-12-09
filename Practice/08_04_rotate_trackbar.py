
import cv2


def on_trackbar(tb_rot):
    global cols, rows, im

    mtx_trf = cv2.getRotationMatrix2D((cols / 2, rows / 2), tb_rot, 1)
    im_trf = cv2.warpAffine(im, mtx_trf, (cols, rows))
    cv2.imshow('im_trf', im_trf)


im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_UNCHANGED)
rows, cols = im.shape[:2]
cv2.imshow('im_trf', im)
cv2.createTrackbar('R', 'im_trf', 0, 360, on_trackbar)

cv2.waitKey(0)
cv2.destroyAllWindows()
