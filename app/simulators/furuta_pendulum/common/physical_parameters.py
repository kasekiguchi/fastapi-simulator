
from dataclasses import dataclass
import numpy as np

@dataclass
class FurutaPendulumParams:
    """
    MATLAB コメントに対応:
    [m1 m2 J jx jy jz L lg Dp Dth gravity a]
    """
    m1: float = 0.22
    m2: float = 0.94
    J: float = 9.1e-3
    jx: float = 0.0095
    jy: float = 0.0095
    jz: float = 0.00017
    L: float = 0.21
    lg: float = 0.35
    Dp: float = 0.0354
    Dth: float = 0.026
    gravity: float = 9.81
    a: float = 3.61

    @property
    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.m1,
                self.m2,
                self.J,
                self.jx,
                self.jy,
                self.jz,
                self.L,
                self.lg,
                self.Dp,
                self.Dth,
                self.gravity,
                self.a,
            ],
            dtype=float,
        )
