# Path: app/furuta_pendulum/FURUTA_PENDULUM/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .apply_input import apply_input
from .measure import measure
from .ode import ode


@dataclass
class FurutaParams:
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


class FURUTA_PENDULUM:
    """
    MATLAB 版 FURUTA_PENDULUM クラスの Python 実装（簡略版）。

    state = [p, th, dp, dth]
      p   : アーム角
      th  : 振り子角
      dp  : p の微分
      dth : th の微分
    """

    def __init__(
        self,
        initial: Sequence[float],
        *,
        sys_noise: float = 1e-5,
        measure_noise: Optional[Sequence[float]] = None,
        params: Optional[FurutaParams] = None,
        dead_zone: float = 0.01,
    ) -> None:
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

        self.params = params or FurutaParams()
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
