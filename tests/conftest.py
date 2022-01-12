from pupil_labs.camera.camera_mpmath import CameraRadial as CameraRadial_MPMath
from pupil_labs.camera.camera_opencv import CameraRadial as CameraRadial_OpenCV
from pupil_labs.camera.camera_scipy import CameraRadial as CameraRadial_SciPy

RADIAL_BACKEND_CLASSES = {
    "mpmath": CameraRadial_MPMath,
    "opencv": CameraRadial_OpenCV,
    "scipy": CameraRadial_SciPy,
}


def pytest_generate_tests(metafunc):
    if "radial_backend_cls" in metafunc.fixturenames:
        metafunc.parametrize("radial_backend_cls", RADIAL_BACKEND_CLASSES.values(), ids=RADIAL_BACKEND_CLASSES.keys())
