from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .apply_input import apply_input
from .measure import measure
from .ode import ode

from ..common.state import FurutaPendulumState
from ..common.physical_parameters import FurutaPendulumParams
from ...base import BaseSimulator, SimState


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
        initial: Optional[Sequence[float]] = None,
        *,
        sys_noise: float = 1e-5,
        # measure_noise: Optional[Sequence[float]] = None,
        measure_noise: float = 1e-3,
        params: Optional[FurutaPendulumParams] = None,
        dead_zone: float = 0.005,
        exp_mode: bool = False,
        param_delta: Optional[Sequence[float]] = None,
    ) -> None:
        self.dt = 0.01
        if initial is None:
            initial = [0.0, 0.0, 0.0, 0.0]
        self.state = np.asarray(initial, dtype=float).copy()  # shape (4,)
        self._exp_sys_noise = float(sys_noise)

        if measure_noise is None:
            # MATLAB: 0.001*[0.005; 2*pi/(2*360)]
            self._exp_measure_noise = 0.01 * np.array(
                [0.005, 2 * np.pi / (2 * 360.0)],
                dtype=float,
            )
        else:
            self._exp_measure_noise = np.asarray(measure_noise, dtype=float)
        self._ideal_measure_noise = np.zeros_like(self._exp_measure_noise)

        self.params = params or FurutaPendulumParams()
        self.param_delta = (
            np.asarray(param_delta, dtype=float)
            if param_delta is not None
            else np.zeros_like(self.params.as_array)
        )
        self.plant_param = self.params.as_array
        self._exp_dead_zone = float(dead_zone)
        self._ideal_dead_zone = 0.0
        self.exp_mode: bool = bool(exp_mode)

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
        self._apply_mode_settings()

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
        self._refresh_plant_params()

    def get_params(self):
        return self.params

    def set_exp_mode(self, enabled: bool) -> None:
        self.exp_mode = bool(enabled)
        self._apply_mode_settings()

    def _apply_mode_settings(self) -> None:
        """Toggle noise, dead-zone, and parameter mismatch based on exp_mode."""
        self.sys_noise = self._exp_sys_noise if self.exp_mode else 0.0
        self.measure_noise = (
            self._exp_measure_noise if self.exp_mode else self._ideal_measure_noise
        )
        self.dead_zone = self._exp_dead_zone if self.exp_mode else self._ideal_dead_zone
        self._refresh_plant_params()

    def _refresh_plant_params(self) -> None:
        delta = self.param_delta if self.exp_mode else 0.0
        self.plant_param = self.params.as_array + delta

    def apply_impulse(self, **kwargs) -> None:
        """クリックなどでトルクを一時的に加えるイメージ"""
        torque = float(kwargs.get("torque", 0.1))
        self.state.u += torque

    def step(self) -> FurutaPendulumState:
        return self.apply_input(self.state.u, self.dt)
