import abc
from pathlib import Path

import numpy as np

from . import types as CT


class Camera(abc.ABC):
    def __init__(
        self,
        pixel_width: int,
        pixel_height: int,
        camera_matrix: CT.CameraMatrix,
        dist_coeffs: CT.DistCoeffs,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pixel_resolution_wh = (pixel_width, pixel_height)

    @property
    def focal_length(self) -> float:
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        return (fx + fy) / 2

    @abc.abstractmethod
    def undistort_image(self, image: CT.Image) -> CT.Image:
        raise NotImplementedError()

    @abc.abstractmethod
    def unproject_points(
        self, points_2d: CT.Points2D, use_distortion: bool = True
    ) -> CT.Points3D:
        raise NotImplementedError()

    @abc.abstractmethod
    def project_points(
        self, points_3d: CT.Points3D, use_distortion: bool = True
    ) -> CT.Points2D:
        raise NotImplementedError()

    def undistort_points_on_image_plane(self, points_2d: CT.Points2D) -> CT.Points2D:
        points_3d = self.unproject_points(points_2d, use_distortion=True)
        points_2d = self.project_points(points_3d, use_distortion=False)
        return points_2d

    def distort_points_on_image_plane(self, points_2d: CT.Points2D) -> CT.Points2D:
        points_3d = self.unproject_points(points_2d, use_distortion=False)
        points_2d = self.project_points(points_3d, use_distortion=True)
        return points_2d

    def save_to_file(self, file_path: Path):
        np.savez(
            file_path, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs
        )

    @classmethod
    def load_from_file(cls, file_path: Path) -> 'Camera':
        data = np.load(file_path)
        return cls(data["camera_matrix"], data["dist_coeffs"])


def CameraRadial(
    pixel_width: int,
    pixel_height: int,
    camera_matrix: CT.CameraMatrix,
    dist_coeffs: CT.DistCoeffs,
    optimization: CT.Optimization,
):

    from .utils import AvailableBackends

    kwargs = {
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
    }

    if optimization == CT.Optimization.ACCURACY and AvailableBackends.has_scipy():
        from .backend_scipy import CameraRadial as CameraRadial_SciPy

        return CameraRadial_SciPy(**kwargs)

    if optimization == CT.Optimization.SPEED and AvailableBackends.has_opencv():
        from .backend_opencv import CameraRadial as CameraRadial_OpenCV

        return CameraRadial_OpenCV(**kwargs)

    from .backend_mpmath import CameraRadial as CameraRadial_MPMath

    return CameraRadial_MPMath(**kwargs)
