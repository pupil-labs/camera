from scipy.fft import idst

from pupil_labs.camera.backend_mpmath import CameraRadial as CameraRadial_MPMath
from pupil_labs.camera.backend_opencv import CameraRadial as CameraRadial_OpenCV
from pupil_labs.camera.backend_scipy import CameraRadial as CameraRadial_SciPy

RADIAL_BACKEND_CLASSES = {
    "mpmath": CameraRadial_MPMath,
    "opencv": CameraRadial_OpenCV,
    "scipy": CameraRadial_SciPy,
}


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="all",
        help="list of backends to test",
    )


def pytest_generate_tests(metafunc):
    if "radial_backend_cls" in metafunc.fixturenames:
        values = []
        ids = []
        for backend in RADIAL_BACKEND_CLASSES:
            selected_backend = metafunc.config.getoption("backend")
            if selected_backend == "all" or backend == selected_backend:
                values.append(RADIAL_BACKEND_CLASSES[backend])
                ids.append(backend)

        metafunc.parametrize("radial_backend_cls", values, ids=ids)
