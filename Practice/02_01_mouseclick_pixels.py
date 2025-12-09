import cv2


def mouse_click(event, x, y, flags, param):
    # Taking global a variable
    global im

    # More mouse events:
    # https://docs.opencv.org/4.12.0/d0/d90/group__highgui__window__flags.html#ga927593befdddc7e7013602bca9b079b0
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print matrix value at click position
        print('Pixel =', im[y, x])
        # If the image has 3 indexable dimensions
        if im.ndim == 3:
            # Printing the red channel value
            print('R =', im[y, x, 2])
            print('')

        # If we change the contents of the image matrix, it must be redisplayed
        cv2.imshow('im', im)


im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_COLOR)
# im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
print('Indexable dimensions of the image:', im.ndim)
print('Image shape:', im.shape)
print('Image pixel type:', im.dtype)

cv2.imshow('im', im)
# Set mouse callback function for the window
cv2.setMouseCallback('im', mouse_click)
# Exit to keypress
cv2.waitKey(0)

cv2.destroyAllWindows()
