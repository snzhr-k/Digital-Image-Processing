"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
https://docs.opencv.org/3.4.1/d9/db0/tutorial_hough_lines.html
"""
import sys
import math
import cv2
import numpy as np


def main(argv):
    # [load]
    default_file = "Sudoku_rs.jpg"
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    im_src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded fine
    if im_src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    # [load]

    # [edge_detection]
    # Edge detection
    im_dst = cv2.Canny(im_src, 50, 200, None, 3)
    # [edge_detection]

    # Copy edges to the images that will display the results in BGR
    im_dst_bgr = cv2.cvtColor(im_dst, cv2.COLOR_GRAY2BGR)
    im_dst_prob_bgr = im_dst_bgr.copy()

    # [hough_lines]
    #  Standard Hough Line Transform
    lines = cv2.HoughLines(im_dst, 1, np.pi / 180, 140, None, 0, 0)
    # [hough_lines]
    # [draw_lines]
    # Diagonal is the longest line that can be drawn
    size_max = math.sqrt(im_dst_bgr.shape[0] ** 2 + im_dst_bgr.shape[1] ** 2)
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # Computing line endpoints outside of image matrix
            pt1 = (int(x0 + size_max * (-b)), int(y0 + size_max * a))
            pt2 = (int(x0 - size_max * (-b)), int(y0 - size_max * a))
            cv2.line(im_dst_bgr, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    # [draw_lines]

    # [hough_lines_p]
    # Probabilistic Line Transform
    lines_prob = cv2.HoughLinesP(im_dst, 1, np.pi / 180, 50, None, 50, 10)
    # [hough_lines_p]
    # [draw_lines_p]
    # Draw the lines
    if lines_prob is not None:
        for i in range(0, len(lines_prob)):
            l = lines_prob[i][0]
            cv2.line(im_dst_prob_bgr, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 3, cv2.LINE_AA)
    # [draw_lines_p]
    # [imshow]
    # Show results
    cv2.imshow("Source", im_src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", im_dst_bgr)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", im_dst_prob_bgr)
    # [imshow]
    # [exit]
    # Wait and Exit
    cv2.waitKey(0)

    return 0
    # [exit]


if __name__ == "__main__":
    main(sys.argv[1:])
