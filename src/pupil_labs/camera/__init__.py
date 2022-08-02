"""Pupil Labs Camera Module"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.camera")
except PackageNotFoundError:
    # package is not installed
    pass


# .version is generated on install via setuptools_scm, see pyproject.toml
from .base import CameraRadial, CameraRadialType, CameraType
from .types import Optimization

__all__ = [
    "__version__",
    "Optimization",
    "CameraRadial",
    "CameraRadialType",
    "CameraType",
]
