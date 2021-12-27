import numpy as np
import scipy.optimize
from . import types as CT
from .camera_mpmath import CameraRadial as Base


class CameraRadial(Base):
    """
    Camera model assuming a lense with radial distortion,
    implemented using SciPy library.
    """
    pass
