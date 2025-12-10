# Path: app/furuta_pendulum/FURUTA_PENDULUM/apply_input.py
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp
from ..common.state import FurutaPendulumState

if TYPE_CHECKING:  # 型チェック用
    from .base import FURUTA_PENDULUM


def apply_input(self: "FURUTA_PENDULUM", u: float, dt: float) -> FurutaPendulumState:
    """
    MATLAB:
        cart.apply_input(u, dt)
    に相当。入力 u を dt 秒間印加し、ODE を1ステップ解いて状態を進める。
    """
    u_eff = u
    if self.dead_zone > 0 and abs(u_eff) < self.dead_zone:
        u_eff = 0.0

    # システムノイズ（MATLAB: u = u + sys_noise*randn/dt）
    if self.sys_noise != 0.0:
        u_eff = u_eff + self.sys_noise * np.random.randn() / dt

    t0 = self.t
    t1 = self.t + dt

    def rhs(t, x):
        return self._ode(x, u_eff, self.plant_param)

    if dt <= 0:
        return self.state

    # RK45 で積分し、失敗時はオイラーでフォールバック
    sol = solve_ivp(
        rhs,
        t_span=(t0, t1),
        y0=self.state,
        t_eval=[t1],  # 終点だけ欲しい
        method="RK45",
        max_step=dt,
        rtol=1e-6,
        atol=1e-9,
    )

    if sol.success and np.all(np.isfinite(sol.y)):
        self.state = sol.y[:, -1]
    else:
        try:
            dx = rhs(t0, self.state)
            if np.all(np.isfinite(dx)):
                self.state = self.state + dt * dx
        except Exception as e:
            print("[FURUTA_PENDULUM] solver failed and euler fallback errored:", e)
            # 状態は据え置き

    self.t = t1
    self.TT.append(self.t)
    self.XX.append(self.state.copy())
    self.output = self._H @ self.state
    return self.state
