import typing

import cv2 as cv
import cv2.typing
import numpy as np


CALIB_PATH = "cam_calib_1920x1080.npz"
with open(CALIB_PATH, "rb") as file:
    npz_file = np.load(file)
    CAMERA_MATRIX = npz_file["mtx"]
    DIST_COEFFS = npz_file["dist_coeff"]
    npz_file.close()


def estimate_pose(
    marker_corners: cv2.typing.MatLike, marker_length_cm: float
) -> typing.Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    # Not using estimate single pose marker since it is deprecated

    # This is the coordinate system of the detection
    # in this case, it is cw top left corner
    obj_points = np.array([
        [-marker_length_cm / 2, marker_length_cm / 2, 0],
        [marker_length_cm / 2, marker_length_cm / 2, 0],
        [marker_length_cm / 2, -marker_length_cm / 2, 0],
        [-marker_length_cm / 2, -marker_length_cm / 2, 0]
    ], np.float32)

    _, rvec, tvec = cv.solvePnP(obj_points, marker_corners, 
                                    CAMERA_MATRIX, DIST_COEFFS)

    return rvec, tvec


def draw_rvec_tvec(
    image: cv2.typing.MatLike, 
    rvec: cv2.typing.MatLike, 
    tvec: cv2.typing.MatLike,
    length: float,
    thickness: int
):
    cv.drawFrameAxes(image, CAMERA_MATRIX, DIST_COEFFS, 
                     rvec, tvec, length, thickness)