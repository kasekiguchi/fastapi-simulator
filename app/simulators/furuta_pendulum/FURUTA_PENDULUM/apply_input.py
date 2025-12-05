# Path: app/furuta_pendulum/FURUTA_PENDULUM/apply_input.py
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:  # 型チェック用
    from .base import FURUTA_PENDULUM


def apply_input(self: "FURUTA_PENDULUM", u: float, dt: float) -> FurutaPendulumState:
    """
    MATLAB:
        cart.apply_input(u, dt)
    に相当。入力 u を dt 秒間印加し、ODE を1ステップ解いて状態を進める。
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
        method="RK45",  # stiff なら "BDF" などに変更可
    )

    self.state = sol.y[:, -1]
    self.t = t1
    self.TT.append(self.t)
    self.XX.append(self.state.copy())
    self.output = self._H @ self.state
    return self.state
