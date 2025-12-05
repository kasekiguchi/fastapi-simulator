
from dataclasses import dataclass
import numpy as np

from ...base import SimState

@dataclass
class FurutaPendulumState(SimState):
    theta: float = 0.0
    dtheta: float = 0.0
    alpha: float = 0.0
    dalpha: float = 0.0
    @property
    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.theta,
                self.alpha,
                self.dtheta,
                self.dalpha
            ])        
    def set(self,state):
        self.theta, self.alpha, self.dtheta, self.dalpha = state
        return self
        