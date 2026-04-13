from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
from scipy.integrate import solve_ivp

from ..base import BaseSimulator, SimState
from .state import TractorTrailerParams, TractorTrailerState
from .reference import TractorTrailerReference, RefSample, HoldReference
from .controller import TractorTrailerController


@dataclass
class PublicTractorTrailerState(SimState):
    """フロントエンドへ送信するトラクタ・トレーラモデルの公開状態"""
    t: float = 0.0
    x1: float = 0.0       # トラクタ後車軸位置（= 連結点）
    y1: float = 0.0
    x2: float = 0.0       # トレーラ後車軸位置
    y2: float = 0.0
    theta1: float = 0.0   # 相対角（tractor - trailer）
    theta2: float = 0.0   # トレーラ絶対姿勢角
    v: float = 0.0
    alpha: float = 0.0
    ref_x: float = 0.0
    ref_y: float = 0.0
    ref_theta: float = 0.0
    # 設計されたフィードバックゲイン（3要素）
    feedback_gain: Optional[list] = None
    # 閉ループ極
    closed_loop_poles: Optional[list] = None


def _tractor_trailer_ode(t, state, v, alpha, L1, L2):
    """トラクタ・トレーラ（運動学モデル）の運動方程式

    独立状態: [x2, y2, theta1, theta2]
      dx2/dt    = v*cos(theta1)*cos(theta2)
      dy2/dt    = v*cos(theta1)*sin(theta2)
      dtheta1/dt = v*tan(alpha)/L1 - v*sin(theta1)/L2
      dtheta2/dt = v*sin(theta1)/L2
    """
    x2, y2, theta1, theta2 = state
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    c2 = math.cos(theta2)
    s2 = math.sin(theta2)
    dx2dt = v * c1 * c2
    dy2dt = v * c1 * s2
    dtheta1dt = v * math.tan(alpha) / L1 - v * s1 / L2
    dtheta2dt = v * s1 / L2
    return [dx2dt, dy2dt, dtheta1dt, dtheta2dt]


class TractorTrailerSimulator(BaseSimulator):
    """トラクタ・トレーラ（運動学）モデルのシミュレータ

    独立状態: [x2, y2, theta1, theta2]
      - (x2, y2): トレーラ後車軸位置
      - theta1 : 相対角 (tractor_heading - trailer_heading)
      - theta2 : トレーラ絶対姿勢角
    派生量（表示用）:
      - (x1, y1) = (x2 + L2*cos(theta2), y2 + L2*sin(theta2))
      - tractor heading phi = theta1 + theta2
    入力: [v, alpha]（トラクタの並進速度と操舵角）
    """

    def __init__(self, dt: float = 0.02, params: Optional[TractorTrailerParams] = None):
        self.dt = dt
        self.params = params or TractorTrailerParams()
        self._initial_state = TractorTrailerState()
        self.state = TractorTrailerState(
            x2=self._initial_state.x2,
            y2=self._initial_state.y2,
            theta1=self._initial_state.theta1,
            theta2=self._initial_state.theta2,
            v=self._initial_state.v,
            alpha=self._initial_state.alpha,
            t=0.0,
        )
        self.controller = TractorTrailerController(self.params)
        self.reference: TractorTrailerReference = HoldReference()
        self._reference_cfg: Optional[Dict[str, Any]] = None
        self.control_mode: str = "controller"  # "controller" | "external"
        self._external_input: tuple[float, float] = (0.0, 0.0)
        self._last_control = (0.0, 0.0)
        self._trace: Dict[str, list] = self._empty_trace()

    def reset(self) -> None:
        """状態を初期値にリセット"""
        self.state = TractorTrailerState(
            x2=self._initial_state.x2,
            y2=self._initial_state.y2,
            theta1=self._initial_state.theta1,
            theta2=self._initial_state.theta2,
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
        x2 = float(initial.get("x2", initial.get("x", 0.0)))
        y2 = float(initial.get("y2", initial.get("y", 0.0)))
        theta1 = float(initial.get("theta1", 0.0))
        theta2 = float(initial.get("theta2", initial.get("theta", 0.0)))
        v = float(initial.get("v", 0.0))
        alpha = float(initial.get("alpha", 0.0))
        self._initial_state = TractorTrailerState(
            x2=x2, y2=y2, theta1=theta1, theta2=theta2, v=v, alpha=alpha, t=0.0,
        )
        self.reset()

    def apply_impulse(self, **kwargs) -> None:
        """外力入力（未使用・no-op）"""
        pass

    def set_params(self, **kwargs) -> None:
        """物理パラメータの変更"""
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))
        # コントローラのゲインも再設計
        self.controller.set_params(self.params)

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
            self.reference = TractorTrailerReference.from_dict(reference)
            self._reference_cfg = reference if isinstance(reference, dict) else None

    def _integrate(self, v: float, alpha: float) -> None:
        """solve_ivp (RK45) で1ステップ積分"""
        L1 = self.params.L1
        L2 = self.params.L2
        y0 = [self.state.x2, self.state.y2, self.state.theta1, self.state.theta2]
        sol = solve_ivp(
            _tractor_trailer_ode,
            [0.0, self.dt],
            y0,
            args=(v, alpha, L1, L2),
            method="RK45",
            dense_output=False,
        )
        x2_new = sol.y[0, -1].item()
        y2_new = sol.y[1, -1].item()
        theta1_new = sol.y[2, -1].item()
        theta2_new = sol.y[3, -1].item()
        self.state = TractorTrailerState(
            x2=x2_new,
            y2=y2_new,
            theta1=theta1_new,
            theta2=theta2_new,
            v=v,
            alpha=alpha,
            t=self.state.t + self.dt,
        )

    def _compute_control(self, ref: RefSample) -> tuple[float, float]:
        """コントローラによる制御入力の計算"""
        if self.controller:
            return self.controller.compute(self.state, ref)
        return ref.v, ref.alpha

    def _compute_control_info(self) -> tuple[Optional[list], Optional[list]]:
        """現在のコントローラから K と閉ループ極を返す（3状態線形化系）"""
        if not self.controller or not hasattr(self.controller, "K"):
            return None, None
        try:
            K = np.asarray(self.controller.K, dtype=float).flatten().tolist()
            A, B = self.controller._get_system_matrices()
            Acl = A - B @ np.asarray(self.controller.K).reshape(1, -1)
            eigs = np.linalg.eigvals(Acl)
            poles = [{"re": float(e.real), "im": float(e.imag)} for e in eigs]
            return K, poles
        except Exception:
            return None, None

    def _tractor_position(self) -> tuple[float, float]:
        """トレーラ状態からトラクタ後車軸（連結点）位置を計算"""
        L2 = self.params.L2
        x1 = self.state.x2 + L2 * math.cos(self.state.theta2)
        y1 = self.state.y2 + L2 * math.sin(self.state.theta2)
        return x1, y1

    def step(self) -> PublicTractorTrailerState:
        """1ステップ進めて公開状態を返す"""
        # x軸時間軸制御: リファレンスは車両の x2 座標に対応する点
        ref = self.reference.sample(self.state.x2) if self.reference else RefSample(np.zeros(2), 0.0, 0.0, 0.0)
        if self.control_mode == "external":
            v_cmd, alpha_cmd = self._external_input
        else:
            v_cmd, alpha_cmd = self._compute_control(ref)
        self._last_control = (v_cmd, alpha_cmd)
        # 1秒ごとにログ出力
        t_now = self.state.t
        if t_now % 1.0 < self.dt or t_now < self.dt:
            print(
                f"[TT] t={t_now:.3f} v={v_cmd:.4f} alpha={alpha_cmd:.4f} "
                f"state=[x2={self.state.x2:.3f},y2={self.state.y2:.3f},"
                f"th1={self.state.theta1:.4f},th2={self.state.theta2:.4f}] "
                f"ref=[{ref.pos[0]:.3f},{ref.pos[1]:.3f},{ref.theta:.4f}]",
                flush=True,
            )
        self._integrate(v_cmd, alpha_cmd)
        self._log_trace(ref)
        K, poles = self._compute_control_info()
        x1, y1 = self._tractor_position()
        return PublicTractorTrailerState(
            t=self.state.t,
            x1=x1,
            y1=y1,
            x2=self.state.x2,
            y2=self.state.y2,
            theta1=self.state.theta1,
            theta2=self.state.theta2,
            v=self.state.v,
            alpha=self.state.alpha,
            ref_x=float(ref.pos[0]),
            ref_y=float(ref.pos[1]),
            ref_theta=float(ref.theta),
            feedback_gain=K,
            closed_loop_poles=poles,
        )

    def get_public_state(self) -> PublicTractorTrailerState:
        """現在の公開状態を返す（ステップを進めない）"""
        ref = self.reference.sample(self.state.x2) if self.reference else RefSample(np.zeros(2), 0.0, 0.0, 0.0)
        x1, y1 = self._tractor_position()
        return PublicTractorTrailerState(
            t=self.state.t,
            x1=x1,
            y1=y1,
            x2=self.state.x2,
            y2=self.state.y2,
            theta1=self.state.theta1,
            theta2=self.state.theta2,
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
        return {
            "t": [], "x1": [], "y1": [], "x2": [], "y2": [],
            "theta1": [], "theta2": [], "v": [], "alpha": [], "ref": [],
        }

    def _log_trace(self, ref: RefSample) -> None:
        """現在の状態をトレースに記録"""
        x1, y1 = self._tractor_position()
        self._trace["t"].append(self.state.t)
        self._trace["x1"].append(x1)
        self._trace["y1"].append(y1)
        self._trace["x2"].append(self.state.x2)
        self._trace["y2"].append(self.state.y2)
        self._trace["theta1"].append(self.state.theta1)
        self._trace["theta2"].append(self.state.theta2)
        self._trace["v"].append(self.state.v)
        self._trace["alpha"].append(self.state.alpha)
        self._trace["ref"].append(
            {"x": float(ref.pos[0]), "y": float(ref.pos[1]), "theta": float(ref.theta)}
        )
