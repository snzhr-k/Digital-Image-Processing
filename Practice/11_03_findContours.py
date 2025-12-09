import cv2

line_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255)]

im = cv2.imread('contour_test_2.png')
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(im_gray, 127, 255, 0)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

print('Hierarchy:')
print(hierarchy.shape)
print(hierarchy)

print('Number of contours:', len(contours))
for cntrIdx in range(0, len(contours)):
    print(cntrIdx, ':', contours[cntrIdx].shape,
          'Next:', hierarchy[0][cntrIdx][0],
          '; Previous:', hierarchy[0][cntrIdx][1],
          '; Child:', hierarchy[0][cntrIdx][2],
          '; Parent:', hierarchy[0][cntrIdx][3])
    cv2.drawContours(im, contours, cntrIdx, line_colors[cntrIdx % 7], 3, cv2.LINE_4)
    cv2.imshow('Contours', im)
    cv2.waitKey(0)

cv2.destroyAllWindows()
