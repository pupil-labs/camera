"""Top-level entry-point for the camera package"""

# .version is generated on install via setuptools_scm, see pyproject.toml
from .base import Camera, CameraRadial
from .types import Optimization
from .version import __version__, __version_info__
