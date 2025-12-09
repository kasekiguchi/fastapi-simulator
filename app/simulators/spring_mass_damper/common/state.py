from dataclasses import dataclass
import numpy as np

from ...base import SimState


@dataclass
class SmdState(SimState):
    p: float = 0.0
    v: float = 0.0

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.p, self.v], dtype=float)

    def set(self, arr):
        self.p, self.v = arr
        return self
