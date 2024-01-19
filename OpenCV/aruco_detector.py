import cv2 as cv 
import numpy as np

from pose_estimation import estimate_pose, draw_rvec_tvec


DICTIONARY = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
PARAMETERS =  cv.aruco.DetectorParameters()
DETECTOR = cv.aruco.ArucoDetector(DICTIONARY, PARAMETERS)
MARKER_LENGTH_CM = 19.6

FONT = cv.FONT_HERSHEY_SIMPLEX

vid = cv.VideoCapture(0, cv.CAP_V4L)
vid.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

while True:

    ret, frame = vid.read()
    frame = cv.flip(frame, 1)

    # detect markers
    marker_corners, ids, rejected = DETECTOR.detectMarkers(frame)

    # estimate pose
    for i in range(len(marker_corners)):
        marker = marker_corners[i][0]
        id_ = ids[i][0]
        rvec, tvec = estimate_pose(marker, MARKER_LENGTH_CM)

        # draw stuff
        draw_rvec_tvec(frame, rvec, tvec, 10, 2)
        cv.polylines(frame, np.int32(marker_corners), True, (255, 0, 0), 2)

        # calculate center to draw dist and id
        cx = 0
        cy = 0
        for corner in marker:
            cx += corner[0]
            cy += corner[1]
        cx = int(cx / 4)
        cy = int(cy / 4)

        text = (f"id: {id_}; x: {tvec[0][0]:.2f}; "
                f"y: {tvec[1][0]:.2f}; z: {tvec[2][0]:.2f}")
        text_size, _ = cv.getTextSize(text, FONT, 1, 2)

        cv.rectangle(frame, (cx, cy - text_size[1] - 2), 
                     (cx + text_size[0], cy + 2), (0, 0, 0), -1)
        cv.putText(frame, text, (cx, cy), FONT, 1, (0, 255, 0), 2)

    smaller = cv.resize(frame, (1280, 720))
    cv.imshow("image", smaller)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
