
import cv2

im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_COLOR)
cv2.imshow('im', im)

print('Wait until key is pressed, no time limit')
key = cv2.waitKey(0)
if key == 0:
    print('Key with no ASCII value assigned was pressed!')
else:
    print('Pressed key and its code:', chr(key), key)

print('Wait for key press, up to 5 seconds')
key = cv2.waitKey(5000)
if key != -1:
    print('A key was pressed!')
else:
    print('There was no key pressed!')

h, w, d = im.shape
angle = 0
step = 1.0
print('Keyboard-monitoring loop')
print('Rotation direction change: r')
print('Exit: q or ESC')

while True:
    key = cv2.waitKeyEx(100)
    if key == -1:
        # No key pressed, but we can do some background activity
        mtx_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        im_rot = cv2.warpAffine(im, mtx_rot, (w, h))
        cv2.imshow('image', im_rot)
        angle += step
        continue

    if key == 27 or key == ord('q'):
        break

    if key == ord('r'):
        step = -step

    if 0 < key < 255:
        print('Pressed key and its code:', chr(key), key)

cv2.destroyAllWindows()
