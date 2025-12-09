import sys
import cv2
import numpy as np


def main(argv):
    # [load]
    default_file = "hun_coins_rs.jpg"
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    im_src = cv2.imread(filename, cv2.IMREAD_COLOR)

    # Check if image is loaded fine
    if im_src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    # [load]

    # [convert_to_gray]
    # Convert it to gray
    im_gray = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
    # [convert_to_gray]

    # [reduce_noise]
    # Reduce the noise to avoid false circle detection
    im_gray = cv2.medianBlur(im_gray, 5)
    # [reduce_noise]

    # [houghcircles]
    num_rows = im_gray.shape[0]
    circles = cv2.HoughCircles(im_gray, cv2.HOUGH_GRADIENT, 1, num_rows / 16,
                               param1=200, param2=20,
                               minRadius=10, maxRadius=60)
    # [houghcircles]

    # [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            print(center, i[2])
            # circle center
            cv2.circle(im_src, center, 1, (255, 0, 0), 3)
            # circle outline
            radius = i[2]
            cv2.circle(im_src, center, radius, (0, 0, 255), 3)
    # [draw]

    # [display]
    cv2.imshow("detected circles", im_src)
    cv2.waitKey(0)
    # [display]

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
