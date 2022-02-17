import abc
import typing as T
import warnings
from pathlib import Path

import numpy as np

from . import types as CT

CameraType = T.TypeVar("CameraType", bound="CameraBase")
CameraRadialType = T.TypeVar("CameraRadialType", bound="CameraRadialBase")


class CameraBase(abc.ABC):
    def __init__(
        self,
        pixel_width: int,
        pixel_height: int,
        camera_matrix: CT.CameraMatrix,
        dist_coeffs: T.Optional[CT.DistCoeffs] = None,
    ):
        if dist_coeffs is None:
            dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)

        dist_coeffs = np.squeeze(dist_coeffs)
        camera_matrix = np.squeeze(camera_matrix)

        if pixel_width <= 0:
            raise ValueError(
                f"pixel_width should be a positive non-zero integer: {pixel_width}"
            )
        if pixel_height <= 0:
            raise ValueError(
                f"pixel_width should be a positive non-zero integer: {pixel_height}"
            )
        if camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix should have 3x3 shape: {camera_matrix.shape}"
            )
        if len(dist_coeffs.shape) != 1:
            raise ValueError(
                f"dist_coeffs should be a 1-dim array: {dist_coeffs.shape}"
            )
        if dist_coeffs.shape[0] < 5:
            # TODO: Not sure about which lengths for dist_coeffs are valid
            raise ValueError(
                f"dist_coeffs shoudl have at least 5 elements: {dist_coeffs.shape}"
            )

        self.camera_matrix = camera_matrix.tolist()
        self.dist_coeffs = dist_coeffs.tolist()
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height

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


class CameraRadialBase(CameraBase):
    pass


def save_radial(file_path: Path, camera: CameraRadialType):
    _dict = {
        "camera_matrix": camera.camera_matrix,
        "dist_coeffs": camera.dist_coeffs,
        "pixel_height": camera.pixel_height,
        "pixel_width": camera.pixel_width,
    }
    np.savez(file_path, **_dict)


def load_radial(
    file_path: Path, optimization: CT.Optimization = CT.Optimization.SPEED
) -> CameraRadialType:
    _dict = np.load(file_path)
    return CameraRadial(
        pixel_width=_dict["pixel_width"],
        pixel_height=_dict["pixel_height"],
        camera_matrix=_dict["camera_matrix"],
        dist_coeffs=_dict["dist_coeffs"],
    )


def CameraRadial(
    pixel_width: int,
    pixel_height: int,
    camera_matrix: CT.CameraMatrix,
    dist_coeffs: CT.DistCoeffs,
    optimization: CT.Optimization = CT.Optimization.SPEED,
) -> CameraRadialType:

    if not isinstance(optimization, CT.Optimization):
        cls_name = f"{CT.Optimization.__module__}.{CT.Optimization.__name__}"
        raise ValueError(f"optimization must be an instance of {cls_name}")

    from .utils import AvailableBackends

    kwargs = {
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
    }

    if optimization == CT.Optimization.ACCURACY:
        if AvailableBackends.has_scipy():
            from .backend_scipy import CameraRadial as CameraRadial_SciPy

            return CameraRadial_SciPy(**kwargs)
        else:
            warnings.warn("Accuracy optimization requires installing scipy extra.")

    if optimization == CT.Optimization.SPEED:
        if AvailableBackends.has_opencv():
            from .backend_opencv import CameraRadial as CameraRadial_OpenCV

            return CameraRadial_OpenCV(**kwargs)
        else:
            warnings.warn("Accuracy optimization requires installing opencv extra.")

    from .backend_mpmath import CameraRadial as CameraRadial_MPMath

    return CameraRadial_MPMath(**kwargs)
