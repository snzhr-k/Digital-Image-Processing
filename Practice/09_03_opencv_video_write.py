
import cv2
import sys

cap = cv2.VideoCapture('sintel_trailer-480p.mp4')
if not cap.isOpened():
    print('Cannot open video file!')
    sys.exit(-1)

cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_fps = cap.get(cv2.CAP_PROP_FPS)

print('Video frame size: : {}x{}'.format(cap_width, cap_height))
print('FPS:', cap_fps)

# Define the codec and create VideoWriter object

# AVI
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, cap_fps, (cap_width, cap_height))

# MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('video_output.mp4', fourcc, cap_fps, (cap_width, cap_height))

while True:
    # Reading video stream frame by frame
    ret, im_frame = cap.read()

    if not ret:
        break

    # Our operations on the frame come here
    im_gray = cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY)
    im_edges = cv2.Canny(im_gray, 50, 150)
    im_out = cv2.cvtColor(im_edges, cv2.COLOR_GRAY2BGR)
    video_out.write(im_out)

    # Display the resulting frame
    cv2.imshow('frame', im_edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release video resources
cap.release()
video_out.release()
cv2.destroyAllWindows()
