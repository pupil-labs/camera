import typing as T

import numpy as np
from mpmath import mp

from . import types as CT
from . import utils
from .camera_base import Camera as Base


class CameraRadial(Base):
    """
    Camera model assuming a lense with radial distortion,
    implemented using mpmath library.
    """

    def undistort_image(self, image: CT.Image) -> CT.Image:
        raise NotImplementedError()  # FIXME

    def unproject_points(
        self, points_2d: CT.Points2D, use_distortion: bool = True
    ) -> CT.Points3D:
        if len(points_2d) == 0:
            return np.asarray([], dtype=np.float32)

        camera_matrix = self.camera_matrix

        if use_distortion:
            dist_coeffs = self.dist_coeffs
        else:
            dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        if len(dist_coeffs) == 5:
            dist_coeffs += [0.0, 0.0, 0.0]

        def unproject_point(point_2d: T.Tuple[float, float]):
            x, y, z = *point_2d, 1.0
            x = (x - camera_matrix[0][2]) / camera_matrix[0][0]
            y = (y - camera_matrix[1][2]) / camera_matrix[1][1]

            result = mp.findroot(
                lambda x_, y_: utils.apply_distortion_model((x_, y_), dist_coeffs)
                - np.asarray([x, y]),
                (x, y),
                verbose=False,
                multidimensional=True,
                tol=1e-4,
                solver="muller",
            )
            x, y = float(result[0]), float(result[1])
            return x, y, z

        points_3d = [unproject_point(p) for p in points_2d]
        return np.asarray(points_3d, dtype=np.float32)

    def project_points(
        self, points_3d: CT.Points3D, use_distortion: bool = True
    ) -> CT.Points2D:
        if len(points_3d) == 0:
            return np.asarray([], dtype=np.float32)

        camera_matrix = self.camera_matrix

        if use_distortion:
            dist_coeffs = self.dist_coeffs
        else:
            dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        if len(dist_coeffs) == 5:
            dist_coeffs += [0.0, 0.0, 0.0]

        def project_point(point_3d: T.Tuple[float, float, float]):
            x, y, _ = point_3d
            x, y = utils.apply_distortion_model((x, y), dist_coeffs)
            x = x * camera_matrix[0][0] + camera_matrix[0][2]
            y = y * camera_matrix[1][1] + camera_matrix[1][2]
            return x, y

        points_2d = [project_point(p) for p in points_3d]
        return np.asarray(points_2d, dtype=np.float32)
