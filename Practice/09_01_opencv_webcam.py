
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Cannot connect to camera hardware!')
    exit(-1)

while True:
    # Reading video stream frame by frame
    ret, im_frame = cap.read()

    if not ret:
        break

    # Processing frame
    im_frame = cv2.flip(im_frame, 1)  # Flip vertically if working with webcam to act as a mirror
    im_gray = cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY)
    im_edges = cv2.Canny(im_gray, 50, 150)

    # Displaying result
    cv2.imshow('frame', im_edges)
    # Call waitkey() to refresh the image window!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resource
cap.release()
cv2.destroyAllWindows()
