# Path: app/furuta_pendulum/FURUTA_PENDULUM/base.py
from __future__ import annotations

from dataclasses import dataclass
from ...base import BaseSimulator, SimState

from typing import Optional, Sequence

import numpy as np

from .apply_input import apply_input
from .measure import measure
from .ode import ode

from ..common.state import FurutaPendulumState
from ..common.physical_parameters import FurutaPendulumParams


class FURUTA_PENDULUM(BaseSimulator):
    """
    MATLAB 版 FURUTA_PENDULUM クラスの Python 実装（簡略版）。

    state = [th, phi, dth, dphi]
      th  : アーム角
      al  : 振り子角
      dth : p の微分
      dal : th の微分
    """

    def __init__(
        self,
        initial: Sequence[float],
        *,
        sys_noise: float = 1e-5,
        measure_noise: Optional[Sequence[float]] = None,
        params: Optional[FurutaPendulumParams] = None,
        dead_zone: float = 0.01,
    ) -> None:
        self.dt = 0.01
        self.state = np.asarray(initial, dtype=float).copy()  # shape (4,)
        self.sys_noise = sys_noise

        if measure_noise is None:
            # MATLAB: 0.001*[0.005; 2*pi/(2*360)]
            self.measure_noise = 0.001 * np.array(
                [0.005, 2 * np.pi / (2 * 360.0)],
                dtype=float,
            )
        else:
            self.measure_noise = np.asarray(measure_noise, dtype=float)

        self.params = params or FurutaPendulumParams()
        self.plant_param = self.params.as_array
        self.dead_zone = dead_zone

        self.t: float = 0.0
        self.input: float = 0.0  # 直前の入力

        # ログ
        self.TT: list[float] = [0.0]
        self.XX: list[np.ndarray] = [self.state.copy()]

        # 出力 y = [p; th]
        self._H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.output = self._H @ self.state

    # ---- MATLABと同名のメソッドを外部ファイルからバインド ----

    apply_input = apply_input
    measure = measure
    _ode = ode
    
    def reset(self) -> None:
        self.state = FurutaPendulumState()

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))

    def get_params(self):
        return self.params

    def apply_impulse(self, **kwargs) -> None:
        """クリックなどでトルクを一時的に加えるイメージ"""
        torque = float(kwargs.get("torque", 0.1))
        self.state.u += torque

    def step(self) -> FurutaPendulumState:
        return self.apply_input(self.state.u,self.dt)