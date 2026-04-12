from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
from scipy.integrate import solve_ivp

from ..base import BaseSimulator, SimState
from .state import TricycleParams, TricycleState
from .reference import TricycleReference, RefSample, HoldReference
from .controller import TricycleController


@dataclass
class PublicTricycleState(SimState):
    """フロントエンドへ送信する三輪車モデルの公開状態"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    alpha: float = 0.0
    ref_x: float = 0.0
    ref_y: float = 0.0
    ref_theta: float = 0.0


def _tricycle_ode(t, state, v, alpha, L):
    """三輪車モデルの運動方程式

    dx/dt = cos(theta) * v
    dy/dt = sin(theta) * v
    dtheta/dt = v * tan(alpha) / L
    """
    x, y, theta = state
    dxdt = math.cos(theta) * v
    dydt = math.sin(theta) * v
    dthetadt = v * math.tan(alpha) / L
    return [dxdt, dydt, dthetadt]


class TricycleSimulator(BaseSimulator):
    """三輪車（前輪操舵）モデルのシミュレータ

    状態: [x, y, theta] （位置と姿勢角）
    入力: [v, alpha] （並進速度と操舵角）
    運動方程式:
        dx/dt = cos(theta) * v
        dy/dt = sin(theta) * v
        dtheta/dt = v * tan(alpha) / L
    L = ホイールベース（後輪軸中心～前輪）
    """

    def __init__(self, dt: float = 0.02, params: Optional[TricycleParams] = None):
        self.dt = dt
        self.params = params or TricycleParams()
        self._initial_state = TricycleState()
        self.state = TricycleState(
            x=self._initial_state.x,
            y=self._initial_state.y,
            theta=self._initial_state.theta,
            v=self._initial_state.v,
            alpha=self._initial_state.alpha,
            t=0.0,
        )
        self.controller = TricycleController(self.params)
        self.reference: TricycleReference = HoldReference()
        self._reference_cfg: Optional[Dict[str, Any]] = None
        self.control_mode: str = "controller"  # "controller" | "external"
        self._external_input: tuple[float, float] = (0.0, 0.0)
        self._last_control = (0.0, 0.0)
        self._trace: Dict[str, list] = self._empty_trace()

    def reset(self) -> None:
        """状態を初期値にリセット"""
        self.state = TricycleState(
            x=self._initial_state.x,
            y=self._initial_state.y,
            theta=self._initial_state.theta,
            v=self._initial_state.v,
            alpha=self._initial_state.alpha,
            t=0.0,
        )
        self._last_control = (0.0, 0.0)
        self._trace = self._empty_trace()

    def set_initial(self, initial=None, **kwargs) -> None:
        """初期状態を設定"""
        if initial is None and kwargs:
            initial = kwargs
        if not isinstance(initial, dict):
            return
        x = float(initial.get("x", 0.0))
        y = float(initial.get("y", 0.0))
        theta = float(initial.get("theta", 0.0))
        v = float(initial.get("v", 0.0))
        alpha = float(initial.get("alpha", 0.0))
        self._initial_state = TricycleState(x=x, y=y, theta=theta, v=v, alpha=alpha, t=0.0)
        self.reset()

    def apply_impulse(self, **kwargs) -> None:
        """外力入力（三輪車モデルでは未使用・no-op）"""
        pass

    def set_params(self, **kwargs) -> None:
        """物理パラメータの変更"""
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))

    def set_control_params(self, control_params: Optional[Dict[str, Any]] = None) -> None:
        """制御パラメータの設定"""
        if not control_params:
            return
        ctype = control_params.get("type") if isinstance(control_params, dict) else None
        if ctype == "external":
            # 外部入力モード: フロントエンドから直接 v, alpha を指定
            inp = control_params.get("input") if isinstance(control_params, dict) else None
            v, alpha = 0.0, 0.0
            if isinstance(inp, (list, tuple)) and len(inp) >= 2:
                v, alpha = float(inp[0]), float(inp[1])
            elif isinstance(inp, dict):
                v = float(inp.get("v", 0.0))
                alpha = float(inp.get("alpha", 0.0))
            self._external_input = (v, alpha)
            self.control_mode = "external"
            return

        self.control_mode = "controller"
        self.controller.set_control_params(control_params)

    def set_reference(self, reference: Optional[Dict[str, Any]]) -> None:
        """リファレンス軌道の設定"""
        if reference is None:
            self.reference = HoldReference()
            self._reference_cfg = None
        else:
            self.reference = TricycleReference.from_dict(reference)
            self._reference_cfg = reference if isinstance(reference, dict) else None

    def _integrate(self, v: float, alpha: float) -> None:
        """solve_ivp (RK45) で1ステップ積分"""
        L = self.params.L
        y0 = [self.state.x, self.state.y, self.state.theta]
        sol = solve_ivp(
            _tricycle_ode,
            [0.0, self.dt],
            y0,
            args=(v, alpha, L),
            method="RK45",
            dense_output=False,
        )
        x_new = sol.y[0, -1].item()
        y_new = sol.y[1, -1].item()
        theta_new = sol.y[2, -1].item()
        self.state = TricycleState(
            x=x_new,
            y=y_new,
            theta=theta_new,
            v=v,
            alpha=alpha,
            t=self.state.t + self.dt,
        )

    def _compute_control(self, ref: RefSample) -> tuple[float, float]:
        """コントローラによる制御入力の計算"""
        if self.controller:
            return self.controller.compute(self.state, ref)
        return ref.v, ref.alpha

    def step(self) -> PublicTricycleState:
        """1ステップ進めて公開状態を返す"""
        ref = self.reference.sample(self.state.t) if self.reference else RefSample(np.zeros(2), 0.0, 0.0, 0.0)
        if self.control_mode == "external":
            v_cmd, alpha_cmd = self._external_input
        else:
            v_cmd, alpha_cmd = self._compute_control(ref)
        self._last_control = (v_cmd, alpha_cmd)
        self._integrate(v_cmd, alpha_cmd)
        self._log_trace(ref)
        return PublicTricycleState(
            t=self.state.t,
            x=self.state.x,
            y=self.state.y,
            theta=self.state.theta,
            v=self.state.v,
            alpha=self.state.alpha,
            ref_x=float(ref.pos[0]),
            ref_y=float(ref.pos[1]),
            ref_theta=float(ref.theta),
        )

    def get_public_state(self) -> PublicTricycleState:
        """現在の公開状態を返す（ステップを進めない）"""
        ref = self.reference.sample(self.state.t) if self.reference else RefSample(np.zeros(2), 0.0, 0.0, 0.0)
        return PublicTricycleState(
            t=self.state.t,
            x=self.state.x,
            y=self.state.y,
            theta=self.state.theta,
            v=self.state.v,
            alpha=self.state.alpha,
            ref_x=float(ref.pos[0]),
            ref_y=float(ref.pos[1]),
            ref_theta=float(ref.theta),
        )

    def get_trace(self) -> Dict[str, list]:
        """シミュレーション履歴を返す"""
        return {k: v.copy() for k, v in self._trace.items()}

    @staticmethod
    def _empty_trace() -> Dict[str, list]:
        """空のトレース辞書を生成"""
        return {"t": [], "x": [], "y": [], "theta": [], "v": [], "alpha": [], "ref": []}

    def _log_trace(self, ref: RefSample) -> None:
        """現在の状態をトレースに記録"""
        self._trace["t"].append(self.state.t)
        self._trace["x"].append(self.state.x)
        self._trace["y"].append(self.state.y)
        self._trace["theta"].append(self.state.theta)
        self._trace["v"].append(self.state.v)
        self._trace["alpha"].append(self.state.alpha)
        self._trace["ref"].append(
            {"x": float(ref.pos[0]), "y": float(ref.pos[1]), "theta": float(ref.theta)}
        )
