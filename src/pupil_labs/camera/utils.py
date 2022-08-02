import numpy as np


class AvailableBackends:
    @staticmethod
    def has_mpmath() -> bool:
        try:
            import mpmath as _  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def has_opencv() -> bool:
        try:
            import cv2 as _  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def has_scipy() -> bool:
        try:
            import scipy as _  # noqa: F401
            import scipy.optimize as _  # noqa: F401, F811

            return True
        except ImportError:
            return False


def apply_distortion_model(point, dist_coeffs):
    x, y = point
    r = np.linalg.norm([x, y])

    k1, k2, p1, p2, k3, k4, k5, k6 = dist_coeffs

    scale = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
    scale /= 1 + k4 * r**2 + k5 * r**4 + k6 * r**6

    x_dist = scale * x + 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
    y_dist = scale * y + p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y

    return np.asarray([x_dist, y_dist])
