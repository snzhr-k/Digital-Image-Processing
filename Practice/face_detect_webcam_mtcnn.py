
import mtcnn
import cv2

print("[INFO] loading face detector...")
detector = mtcnn.MTCNN()

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)

        print('Videó méret: {}x{}'.format(cap_width, cap_height))
        print('FPS:', cap_fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 4, (cap_width, cap_height))

    while True:
        ret, im = cap.read()
        if not ret:
            break

        # image = cv2.flip(image, 1)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        res = detector.detect_faces(im)
        # Returned structure example:
        # [
        #     {
        #         'box': [277, 90, 48, 63],
        #         'keypoints':
        #         {
        #             'nose': (303, 131),
        #             'mouth_right': (313, 141),
        #             'right_eye': (314, 114),
        #             'left_eye': (291, 117),
        #             'mouth_left': (296, 143)
        #         },
        #         'confidence': 0.99851983785629272
        #     }
        # ]

        rects = []
        points = []
        for i in range(len(res)):
            # print(res[i]['box'])
            rects.append(res[i]['box'])
            points.append(res[i]['keypoints']['right_eye'])
            points.append(res[i]['keypoints']['left_eye'])
            points.append(res[i]['keypoints']['nose'])
            points.append(res[i]['keypoints']['mouth_right'])
            points.append(res[i]['keypoints']['mouth_left'])

        # loop over the bounding boxes
        for (x, y, w, h) in rects:
            # draw the face bounding box on the image
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # loop over points
        for (x, y) in points:
            # draw the points on the image
            cv2.circle(im, (x, y), 3, (255, 255, 0), 2)

        # show the output image
        cv2.imshow("Image", im)
        out.write(im)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            cap.release()
            out.release()
            break


cv2.destroyAllWindows()
