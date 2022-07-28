import enum

from numpy import float32, uint8  # noqa
from numpy.typing import NDArray  # noqa
from typing_extensions import TypeAlias

CameraMatrix: TypeAlias = "NDArray[float32]"
DistCoeffs: TypeAlias = "NDArray[float32]"
Image: TypeAlias = "NDArray[uint8]"
Points2D: TypeAlias = "NDArray[float32]"
Points3D: TypeAlias = "NDArray[float32]"


class AutoLowercaseNameEnum(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Optimization(AutoLowercaseNameEnum):
    ACCURACY = enum.auto()
    SPEED = enum.auto()
