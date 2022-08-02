import contextlib

import pytest

from pupil_labs.camera import CameraRadial, CameraRadialType, Optimization
from pupil_labs.camera.utils import AvailableBackends


@contextlib.contextmanager
def custom_import():
    """
    https://stackoverflow.com/a/47854417
    """

    import importlib

    allowed_modules = dict()

    def restricted_importer(name, globals=None, locals=None, fromlist=(), level=0):

        if name in allowed_modules and allowed_modules[name] is False:
            raise ImportError(f"module {name} is not allowed.")

        # not exactly a good verification layer
        frommodule = globals['__name__'] if globals else None

        if (
            frommodule
            and frommodule in allowed_modules
            and allowed_modules[frommodule] is False
        ):
            raise ImportError(f"module {frommodule} is not allowed.")

        return importlib.__import__(name, globals, locals, fromlist, level)

    default_importer = __builtins__['__import__']
    assert default_importer
    __builtins__['__import__'] = restricted_importer

    try:
        yield allowed_modules
    finally:
        __builtins__['__import__'] = default_importer


def camera_for_optimization(optimization: Optimization) -> CameraRadialType:
    return CameraRadial(
        1, 1, [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0, 0, 0], optimization
    )


def test_backend_imports():
    with custom_import() as _:

        assert AvailableBackends.has_mpmath(), "MPMath backend should be available"

        assert AvailableBackends.has_opencv(), "OpenCV backend should be available"

        assert AvailableBackends.has_scipy(), "SciPy backend should be available"


def test_backend_imports_no_cv2():
    with custom_import() as allowed_modules:
        allowed_modules["cv2"] = False
        allowed_modules["scipy"] = True

        assert AvailableBackends.has_mpmath(), "MPMath backend should be available"

        assert AvailableBackends.has_scipy(), "SciPy backend should be available"

        assert (
            not AvailableBackends.has_opencv()
        ), "OpenCV backend should NOT be available"


def test_backend_imports_no_scipy():
    with custom_import() as allowed_modules:
        allowed_modules["cv2"] = True
        allowed_modules["scipy"] = False

        assert AvailableBackends.has_mpmath(), "MPMath backend should be available"

        assert AvailableBackends.has_opencv(), "OpenCV backend should be available"

        assert (
            not AvailableBackends.has_scipy()
        ), "SciPy backend should NOT be available"


def test_backend_imports_no_cv2_no_scipy():
    with custom_import() as allowed_modules:
        allowed_modules["cv2"] = False
        allowed_modules["scipy"] = False

        assert AvailableBackends.has_mpmath(), "MPMath backend should be available"

        assert (
            not AvailableBackends.has_opencv()
        ), "OpenCV backend should NOT be available"

        assert (
            not AvailableBackends.has_scipy()
        ), "SciPy backend should NOT be available"


def test_backend_selection_base():
    with custom_import() as allowed_modules:
        allowed_modules["cv2"] = False
        allowed_modules["scipy"] = False

        from pupil_labs.camera.backend_mpmath import CameraRadial as CameraRadial_MPMath

        optimization = Optimization.ACCURACY
        with pytest.warns(UserWarning):
            camera = camera_for_optimization(optimization)
        assert isinstance(camera, CameraRadial_MPMath)

        optimization = Optimization.SPEED
        with pytest.warns(UserWarning):
            camera = camera_for_optimization(optimization)
        assert isinstance(camera, CameraRadial_MPMath)


def test_backend_selection_with_opencv():
    with custom_import() as allowed_modules:
        allowed_modules["cv2"] = True
        allowed_modules["scipy"] = False

        from pupil_labs.camera.backend_mpmath import CameraRadial as CameraRadial_MPMath
        from pupil_labs.camera.backend_opencv import CameraRadial as CameraRadial_OpenCV

        optimization = Optimization.ACCURACY
        with pytest.warns(UserWarning):
            camera = camera_for_optimization(optimization)
        assert isinstance(camera, CameraRadial_MPMath)

        optimization = Optimization.SPEED
        camera = camera_for_optimization(optimization)
        assert isinstance(camera, CameraRadial_OpenCV)


def test_backend_selection_with_scipy():
    with custom_import() as allowed_modules:
        allowed_modules["cv2"] = False
        allowed_modules["scipy"] = True

        from pupil_labs.camera.backend_mpmath import CameraRadial as CameraRadial_MPMath
        from pupil_labs.camera.backend_scipy import CameraRadial as CameraRadial_SciPy

        optimization = Optimization.ACCURACY
        camera = camera_for_optimization(optimization)
        assert isinstance(camera, CameraRadial_SciPy)

        optimization = Optimization.SPEED
        with pytest.warns(UserWarning):
            camera = camera_for_optimization(optimization)
        assert isinstance(camera, CameraRadial_MPMath)


def test_backend_selection_with_opencv_with_scipy():
    with custom_import() as allowed_modules:
        allowed_modules["cv2"] = True
        allowed_modules["scipy"] = True

        from pupil_labs.camera.backend_opencv import CameraRadial as CameraRadial_OpenCV
        from pupil_labs.camera.backend_scipy import CameraRadial as CameraRadial_SciPy

        optimization = Optimization.ACCURACY
        camera = camera_for_optimization(optimization)
        assert isinstance(camera, CameraRadial_SciPy)

        optimization = Optimization.SPEED
        camera = camera_for_optimization(optimization)
        assert isinstance(camera, CameraRadial_OpenCV)
