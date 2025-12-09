from dataclasses import dataclass
import numpy as np


@dataclass
class SmdParams:
    mass: float = 1.0
    k: float = 1.0
    c: float = 0.1

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.mass, self.k, self.c], dtype=float)
