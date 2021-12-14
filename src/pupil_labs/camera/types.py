import nptyping as npt
import typing as T

CameraMatrix = npt.NDArray[(3, 3), npt.Float[32]]
DistCoeffs = npt.NDArray[T.Any, npt.Float[32]]
Image = npt.NDArray[(T.Any, T.Any), npt.UInt[8]]
Points2D = npt.NDArray[(T.Any, 2), npt.Float[32]]
Points3D = npt.NDArray[(T.Any, 3), npt.Float[32]]
