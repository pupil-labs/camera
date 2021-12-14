import nptyping as npt
import typing as T

CameraMatrix = npt.NDArray[3, 3, npt.Float[32]]
DistCoeffs = npt.NDArray[T.Any, npt.Float[32]]
Image = npt.NDArray  #TODO: Provide dimensions
Points2D = npt.NDArray  #TODO: Provide dimensions
Points3D = npt.NDArray  #TODO: Provide dimensions
