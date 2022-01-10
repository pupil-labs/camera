import cv2
import numpy as np

from . import types as CT
from .camera_mpmath import CameraRadial as Base


class CameraRadial(Base):
    """
    Camera model assuming a lense with radial distortion,
    implemented using OpenCV library.
    """

    def undistort_image(self, image: CT.Image) -> CT.Image:
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def unproject_points(
        self, points_2d: CT.Points2D, use_distortion: bool = True
    ) -> CT.Points3D:
        if use_distortion:
            dist_coeffs = self.dist_coeffs
        else:
            dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)
        camera_matrix = np.asarray(self.camera_matrix, dtype=np.float32)

        points_2d = np.asarray(points_2d, dtype=np.float32)
        points_2d = points_2d.reshape((-1, 1, 2))
        points_2d = cv2.undistortPoints(points_2d, camera_matrix, dist_coeffs)

        points_3d = cv2.convertPointsToHomogeneous(points_2d)
        points_3d = points_3d.reshape((-1, 3))
        return points_3d

    def project_points(
        self, points_3d: CT.Points3D, use_distortion: bool = True
    ) -> CT.Points2D:
        if use_distortion:
            dist_coeffs = self.dist_coeffs
        else:
            dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        rvec = np.zeros((1, 1, 3))
        tvec = np.zeros((1, 1, 3))

        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)
        camera_matrix = np.asarray(self.camera_matrix, dtype=np.float32)

        points_3d = np.asarray(points_3d, dtype=np.float32)
        points_3d = points_3d.reshape((1, -1, 3))

        points_2d, _ = cv2.projectPoints(
            points_3d, rvec, tvec, camera_matrix, dist_coeffs
        )
        points_2d = points_2d.reshape((-1, 2))
        return points_2d
