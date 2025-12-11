from dataclasses import dataclass

from ..base import SimState


@dataclass
class DiffDriveState(SimState):
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0


@dataclass
class DiffDriveParams:
    L: float = 0.5  # tread length
