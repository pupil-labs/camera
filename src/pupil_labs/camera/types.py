import enum
from typing import Any  # noqa

from nptyping import Float32, NDArray, Shape, UInt8  # noqa
from typing_extensions import TypeAlias

CameraMatrix: TypeAlias = "NDArray[Shape['3, 3'], Float32]"
DistCoeffs: TypeAlias = "NDArray[Any, Float32]"
Image: TypeAlias = "NDArray[Shape[Any, Any], UInt8]"
Points2D: TypeAlias = "NDArray[Shape[Any, 2], Float32]"
Points3D: TypeAlias = "NDArray[Shape[Any, 3], Float32]"


class AutoLowercaseNameEnum(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Optimization(AutoLowercaseNameEnum):
    ACCURACY = enum.auto()
    SPEED = enum.auto()
