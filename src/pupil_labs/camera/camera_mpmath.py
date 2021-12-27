from mpmath import mp
from . import utils
from . import types as CT
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
        raise NotImplementedError()  # FIXME

    def project_points(
        self, points_3d: CT.Points3D, use_distortion: bool = True
    ) -> CT.Points2D:
        if len(points_3d) == 0:
            return []

        camera_matrix = self.camera_matrix
        dist_coeffs = self.dist_coeffs

        if len(points_3d) == 5:
            dist_coeffs += [0.0, 0.0, 0.0]

        def project_point(point):
            x_, y_ = utils.apply_distortion_model(point, dist_coeffs)
            x_ = x_ * camera_matrix[0][0] + camera_matrix[0][2]
            y_ = y_ * camera_matrix[1][1] + camera_matrix[1][2]
            return x_, y_

        return list(map(project_point, points_3d))
