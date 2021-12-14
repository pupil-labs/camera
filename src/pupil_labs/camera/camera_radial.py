import cv2
import numpy as np
from . import types as CT
from .camera_base import CameraABC


class CameraRadial(CameraABC):
    """
    Camera model assuming a lense with radial distortion.
    """

    def undistort_image(self, image: CT.Image) -> CT.Image:
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def unproject_points(self, points_2d: CT.Points2D, use_distortion: bool = True) -> CT.Points3D:
        if use_distortion:
            dist_coeffs = self.dist_coeffs
        else:
            dist_coeffs = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]])

        points_2d = points_2d.reshape((-1, 1, 2))
        points_2d = cv2.undistortPoints(points_2d, self.camera_matrix, dist_coeffs)

        points_3d = cv2.convertPointsToHomogeneous(points_2d)
        points_3d = points_3d.reshape((-1, 3))

        return points_3d

    def project_points(self, points_3d: CT.Points3D, use_distortion: bool = True) -> CT.Points2D:
        if use_distortion:
            dist_coeffs = self.dist_coeffs
        else:
            dist_coeffs = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]])

        rvec = np.zeros(3).reshape(1, 1, 3)
        tvec = np.zeros(3).reshape(1, 1, 3)

        points_3d = points_3d.reshape((1, -1, 3))

        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, self.camera_matrix, dist_coeffs)
        points_2d = points_2d.reshape((-1, 2))

        return points_2d
