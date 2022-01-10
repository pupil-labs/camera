import numpy as np
from numpy.testing import assert_almost_equal

from pupil_labs.camera.camera_mpmath import CameraRadial as CameraRadial_MPMath
from pupil_labs.camera.camera_opencv import CameraRadial as CameraRadial_OpenCV
from pupil_labs.camera.camera_scipy import CameraRadial as CameraRadial_SciPy

PIXEL_WIDTH = 1088
PIXEL_HEIGHT = 1080
CAMERA_MATRIX = [
    [766.3037717610379, 0.0, 559.7158729463123],
    [0.0, 765.4514012936911, 537.2187571096966],
    [0.0, 0.0, 1.0],
]
DIST_COEFFS = [
    -0.12571787111434657,
    0.1009174721106796,
    0.0004064475713640723,
    -0.0001776950802199194,
    0.017309286074375808,
    0.20449589859897552,
    0.008640898256976831,
    0.06428433887310138,
]
CAMERA_RADIAL_KS = [
    CameraRadial_MPMath,
    CameraRadial_OpenCV,
    CameraRadial_SciPy,
]


def test_unproject_points():
    points_2d = [[100, 200], [800, 600]]
    points_3d = [[-0.75240, -0.55311, 1.0], [0.32508, 0.08498, 1.0]]

    for cls in CAMERA_RADIAL_KS:
        camera = cls(
            pixel_width=PIXEL_WIDTH,
            pixel_height=PIXEL_HEIGHT,
            camera_matrix=CAMERA_MATRIX,
            dist_coeffs=DIST_COEFFS,
        )
        assert_almost_equal(
            camera.unproject_points(points_2d, use_distortion=True),
            np.asarray(points_3d),
            decimal=3,
        )


def test_project_points():
    points_2d = [[100.3349, 200.2458], [799.9932, 599.9996]]
    points_3d = [[-0.75170, -0.55260, 1.0], [0.32508, 0.08498, 1.0]]

    for cls in CAMERA_RADIAL_KS:
        camera = cls(
            pixel_width=PIXEL_WIDTH,
            pixel_height=PIXEL_HEIGHT,
            camera_matrix=CAMERA_MATRIX,
            dist_coeffs=DIST_COEFFS,
        )
        assert_almost_equal(
            camera.project_points(points_3d, use_distortion=True),
            np.asarray(points_2d),
            decimal=4,
        )
