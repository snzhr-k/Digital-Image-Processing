
import cv2
import sys

cap = cv2.VideoCapture('sintel_trailer-480p.mp4')

if not cap.isOpened():
    print('Cannot open video file!')
    sys.exit(-1)

cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_fps = cap.get(cv2.CAP_PROP_FPS)

print('Video frame size: {}x{}'.format(cap_width, cap_height))
print('FPS:', cap_fps)

while True:
    # Reading video stream frame by frame
    ret, im_frame = cap.read()

    if not ret:
        break

    # Processing frame
    im_gray = cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY)
    im_edges = cv2.Canny(im_gray, 50, 150)

    # Displaying result
    cv2.imshow('frame', im_edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resource
cap.release()
cv2.destroyAllWindows()
