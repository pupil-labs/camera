import enum

from numpy import float32, uint8
from numpy.typing import NDArray

CameraMatrix = NDArray[float32]
DistCoeffs = NDArray[float32]
Image = NDArray[uint8]
Points2D = NDArray[float32]
Points3D = NDArray[float32]


class AutoLowercaseNameEnum(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Optimization(AutoLowercaseNameEnum):
    ACCURACY = enum.auto()
    SPEED = enum.auto()
