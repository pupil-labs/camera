import numpy as np
import pytest
from numpy.testing import assert_almost_equal

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


@pytest.mark.parametrize(
    'points',
    [
        np.array([(100, 200), (800, 600)]),  # unstructured ints
        np.array(
            [(100, 200), (800, 600)],
            dtype=[('x', np.int32), ('y', np.int32)],
        ),  # structured ints
        np.array([(100, 200), (800, 600)], dtype=np.int32),  # unstructured ints
        np.array(
            [(100.0, 200.0), (800.0, 600.0)],
            dtype=[('x', np.float32), ('y', np.float32)],
        ),  # structured floats
        np.array(
            [(100.0, 200.0), (800.0, 600.0)],
            dtype=[('x', np.float32), ('y', np.float32)],
        ),  # structured floats
        [(100, 200), (800, 600)],  # list of tuples
        ([100, 200], [800, 600]),  # tuple of lists
        [[100, 200], [800, 600]],  # list of lists
        ((100, 200), (800, 600)),  # tuple of tuples
    ],
)
def test_unproject_points(radial_backend_cls, points):
    expected = np.array([[-0.75240, -0.55311, 1.0], [0.32508, 0.08498, 1.0]])

    camera = radial_backend_cls(
        pixel_width=PIXEL_WIDTH,
        pixel_height=PIXEL_HEIGHT,
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
    )
    assert_almost_equal(
        camera.unproject_points(points, use_distortion=True),
        np.asarray(expected),
        decimal=3,
    )


@pytest.mark.parametrize(
    'points',
    [
        np.array([(-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)]),  # unstructured
        np.array(
            [(-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)],
            dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)],
        ),  # structured
        [(-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)],  # list of tuples
        ([-0.75170, -0.55260, 1.0], [0.32508, 0.08498, 1.0]),  # tuple of lists
        [[-0.75170, -0.55260, 1.0], [0.32508, 0.08498, 1.0]],  # list of lists
        ((-0.75170, -0.55260, 1.0), (0.32508, 0.08498, 1.0)),  # tuple of tuples
    ],
)
def test_project_points(radial_backend_cls, points):
    expected = np.array([(100.3349, 200.2458), (799.9932, 599.9996)])
    camera = radial_backend_cls(
        pixel_width=PIXEL_WIDTH,
        pixel_height=PIXEL_HEIGHT,
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
    )
    assert_almost_equal(
        camera.project_points(points, use_distortion=True),
        expected,
        decimal=4,
    )
