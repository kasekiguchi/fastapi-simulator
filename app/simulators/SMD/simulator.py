# app/simulators/SMD/simulator.py
from dataclasses import dataclass
from ..base import BaseSimulator, SimState


@dataclass
class SmdParams:
    mass: float = 1.0
    k: float = 1.0
    c: float = 0.1


@dataclass
class SmdState(SimState):
    x: float = 0.0
    v: float = 0.0
    ext: float = 0.0  # 外力


class SmdSimulator(BaseSimulator):
    """1自由度 質点バネダンパモデル"""

    def __init__(self, dt: float = 0.01):
        self.params = SmdParams()
        self.state = SmdState()
        self.dt = dt

    def reset(self) -> None:
        self.state = SmdState()

    def set_params(self, **kwargs) -> None:
        # mass, k, c を更新
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))

    def apply_impulse(self, **kwargs) -> None:
        """force [N] を1ステップだけ加える"""
        force = float(kwargs.get("force", 1.0))
        self.state.ext += force

    def step(self) -> SmdState:
        p = self.params
        s = self.state

        # m x'' + c x' + k x = F
        a = (s.ext - p.c * s.v - p.k * s.x) / p.mass
        s.v += a * self.dt
        s.x += s.v * self.dt
        s.t += self.dt

        # 外力は1ステップで消費
        s.ext = 0.0

        return s
