import typing as T

import numpy as np
import scipy.optimize

from . import types as CT
from . import utils
from .backend_mpmath import CameraRadial as Base


class CameraRadial(Base):
    """
    Camera model assuming a lense with radial distortion,
    implemented using SciPy library.
    """

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

            result = scipy.optimize.root(
                lambda p_norm: utils.apply_distortion_model(p_norm, dist_coeffs)
                - np.asarray([x, y]),
                [[x, y]],
                method="lm",
            )
            x, y = result.x[0], result.x[1]

            return x, y, z

        points_3d = [unproject_point(p) for p in points_2d]
        return np.asarray(points_3d, dtype=np.float32)
