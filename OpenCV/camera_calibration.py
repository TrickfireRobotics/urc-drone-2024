# Based off https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

from os import path
import time

import cv2 as cv
import numpy as np


# termination criteria
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
ROW_COUNT = 8
COL_COUNT = 6
SQUARE_SIZE_CM = 3
objp = np.zeros((COL_COUNT * ROW_COUNT, 3), np.float32)
grid = np.mgrid[0:ROW_COUNT * SQUARE_SIZE_CM:SQUARE_SIZE_CM,
                0:COL_COUNT * SQUARE_SIZE_CM:SQUARE_SIZE_CM]
objp[:,:2] = grid.T.reshape(-1,2)

# Arrays to store object points and image points from all the images.'
CAPTURE_INTERVAL_SECS = 2
last_capture_time = time.time()
obj_points = [] # 3d point in real world space
img_points = [] # 2d points in image plane.

FONT = cv.FONT_HERSHEY_SIMPLEX
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
vid = cv.VideoCapture(0, cv.CAP_V4L)
vid.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
SCALE = (CAM_WIDTH / PROCESS_WIDTH, CAM_HEIGHT / PROCESS_HEIGHT)

print(("calibrator started. press \"q\" at any time to finish capturing and "
       "start calibration."))

while True:
    ret, frame = vid.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_small = cv.resize(gray, (PROCESS_WIDTH, PROCESS_HEIGHT),
                           cv.INTER_CUBIC)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray_small, 
                                            (ROW_COUNT, COL_COUNT), None)

    # If found, add object points, image points (after refining them)
    if ret:
        # rescale the output of findChessboardCorners
        corners *= SCALE
    
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

        curr_time = time.time()
        if curr_time - last_capture_time >= CAPTURE_INTERVAL_SECS:
            last_capture_time = curr_time
            obj_points.append(objp)
            img_points.append(corners2)
            print(f"capture taken. currently {len(obj_points)} captures")

        # Draw and display the corners
        cv.drawChessboardCorners(frame, (ROW_COUNT, COL_COUNT), corners2, ret)
    
    # Print number of captures taken
    cv.putText(frame, f"captures: {len(obj_points)}", 
               (10, 50), FONT, 1, (0, 255, 0))
    cv.imshow('img', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()

ret, mtx, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, 
                                                  gray.shape[::-1], None, None)

print("results:")
print("------------------------------------------------------------------")
print(f"{'return value = ':<26}{ret}")
print(f"{'camera matrix = ':<26}\n{mtx}\n")
print(f"{'distortion coefficients = ':<26}\n{dist_coeffs}\n")
print(f"{'rvecs = ':<26}\n{rvecs}\n")
print(f"{'tvecs = ':<26}\n{tvecs}\n")

mean_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, 
                                     dist_coeffs)
    error = cv.norm(img_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print(f"{'reprojection error =':<26} {mean_error / len(obj_points)}")

invalid = True
file_name = f"cam_calib_{CAM_WIDTH}x{CAM_HEIGHT}.npz"
while invalid:
    save_path = input("please enter the save folder: ")
    try:
        with open(path.join(save_path, file_name), "wb") as file:
            np.savez(file, mtx=mtx, dist_coeff=dist_coeffs)
        invalid = False
    except:
        print("path invalid")
    