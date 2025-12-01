# Path: app/furuta_pendulum/FURUTA_PENDULUM.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp


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


class FurutaPendulum:
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

        self.t = 0.0
        self.input = 0.0  # 直前の入力
        self.TT = [0.0]
        self.XX = [self.state.copy()]

        # 出力 y = [p; th]
        self._H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.output = self._H @ self.state

    # --------- MATLAB と同名 API ---------

    def apply_input(self, u: float, dt: float) -> None:
        """
        MATLAB:
            cart.apply_input(u, dt)
        に相当。入力 u を dt 秒間印加し、ODE を解いて状態を進める。
        """
        # デッドゾーン
        if abs(u) < self.dead_zone:
            u_eff = 0.0
        else:
            u_eff = u

        # システムノイズ（MATLAB: u = u + sys_noise*randn/dt）
        u_eff = u_eff + self.sys_noise * np.random.randn() / dt

        t0 = self.t
        t1 = self.t + dt

        def rhs(t, x):
            return self._ode(x, u_eff, self.plant_param)

        sol = solve_ivp(
            rhs,
            t_span=(t0, t1),
            y0=self.state,
            t_eval=[t1],  # 終点だけ欲しい
            method="RK45",
        )

        self.state = sol.y[:, -1]
        self.t = t1
        self.TT.append(self.t)
        self.XX.append(self.state.copy())
        self.output = self._H @ self.state

    def measure(self, t: Optional[float] = None) -> np.ndarray:
        """
        MATLAB:
            output = obj.measure(t);
        の簡易版。ここでは t 無視で「現在の状態」を測定する。
        本気で一致させたい場合は TT, XX から補間しても良い。
        """
        self.output = self._H @ self.state
        noise = self.measure_noise * np.random.randn(2)
        return self.output + noise

    # --------- 非公開: 連続時間 ODE ---------

    def _ode(self, x: np.ndarray, u: float, param: np.ndarray) -> np.ndarray:
        """
        MATLAB の private メソッド ode(obj,in1,input,in3) に相当。
        ここに、Symbolic で生成された式を Python に移植して貼る。

        いったんは簡易モデル（線形っぽいもの）にしておき、
        接続や UI を先に完成させる想定。
        """
        p, th, dp, dth = x
        (
            m1,
            m2,
            J,
            jx,
            jy,
            jz,
            L,
            lg,
            Dp,
            Dth,
            gravity,
            a,
        ) = param

        # TODO: ここに MATLAB の t2=cos(p) ... の長い式を移植する
        # --- 今はダミーの安定化モデル ---
        k_p = 1.0
        k_th = 1.0

        ddp = -k_p * p + a * u - Dp * dp
        ddth = -k_th * th - Dth * dth + gravity * np.sin(th) * 0.0  # 適当

        return np.array([dp, dth, ddp, ddth], dtype=float)
