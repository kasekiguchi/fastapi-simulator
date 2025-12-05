
from dataclasses import dataclass
import numpy as np

from ...base import SimState

@dataclass
class FurutaPendulumState(SimState):
    theta: float = 0.0
    dtheta: float = 0.0
    phi: float = 0.0
    dphi: float = 0.0
    @property
    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.theta,
                self.phi,
                self.dtheta,
                self.dphi
            ])        
    def set(self,state):
        self.theta, self.phi, self.dtheta, self.dphi = state
        return self
        