from __future__ import annotations

import math
from typing import Dict, Any, List

import numpy as np
from scipy import signal, linalg
from ..common.linear_utils import ackermann

from .reference import RefSample
from .state import TractorTrailerState, TractorTrailerParams


def _wrap_angle(a: float) -> float:
    """角度を [-pi, pi) に正規化"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _parse_poles(raw: List) -> list[complex]:
    """フロントから送られる極のリストをパース"""
    parsed = []
    for p in raw:
        if isinstance(p, dict):
            parsed.append(complex(float(p.get("re", 0.0)), float(p.get("im", 0.0))))
        else:
            try:
                parsed.append(complex(p))
            except Exception:
                continue
    return parsed


def _design_K_pole(A: np.ndarray, B: np.ndarray, poles_raw: List) -> tuple[np.ndarray, str | None]:
    """極配置でゲイン K を設計（連続系）"""
    poles = _parse_poles(poles_raw)
    nx = A.shape[0]
    if len(poles) < nx:
        poles.extend([complex(-1.0 * (i + 1), 0) for i in range(nx - len(poles))])
    poles = poles[:nx]
    for p in list(poles):
        if abs(p.imag) > 0 and np.conj(p) not in poles:
            poles.append(np.conj(p))
    poles = poles[:nx]
    # 1. place_poles (YT法)
    try:
        res = signal.place_poles(A, B, poles)
        K = np.asarray(res.gain_matrix, dtype=float).flatten()
        print(f"[TT] pole placement (YT): poles={poles} -> K={K}", flush=True)
        return K, None
    except Exception as e:
        print(f"[TT] YT failed: {e}", flush=True)
    # 2. Ackermann法（重極対応）
    try:
        K = ackermann(A, B, poles).flatten()
        print(f"[TT] pole placement (Ackermann): poles={poles} -> K={K}", flush=True)
        return K, None
    except Exception as e:
        print(f"[TT] Ackermann failed: {e}", flush=True)
        return np.zeros(nx, dtype=float), str(e)


def _design_K_lqr(A: np.ndarray, B: np.ndarray, Q_vals: List, R_vals: List) -> tuple[np.ndarray, str | None]:
    """LQR でゲイン K を設計（連続系）"""
    nx = A.shape[0]
    nu = B.shape[1]
    Q_arr = np.asarray(Q_vals, dtype=float).flatten()
    R_arr = np.asarray(R_vals, dtype=float).flatten()
    Q = np.diag(Q_arr[:nx]) if Q_arr.size >= nx else np.eye(nx)
    R = np.diag(R_arr[:nu]) if R_arr.size >= nu else np.eye(nu)
    try:
        P = linalg.solve_continuous_are(A, B, Q, R)
        K = (np.linalg.inv(R) @ B.T @ P).flatten()
        print(f"[TT] LQR: Q=diag({np.diag(Q)}), R=diag({np.diag(R)}) -> K={K}", flush=True)
        return K, None
    except Exception as e:
        print(f"[TT] LQR CARE failed: {e}", flush=True)
        return np.zeros(nx, dtype=float), str(e)


class TractorTrailerController:
    """トラクタ・トレーラモデルのコントローラ

    制御モード (type):
      - "approximate": 近似線形化による状態フィードバック (3状態: y2, theta2, theta1)
      - "exact": 厳密線形化（x軸時間軸状態制御正準形、3重積分器）

    ゲイン設計法 (design):
      - "manual": 直接 K を指定
      - "pole_assignment": 極配置法（Ackermann対応）
      - "lqr": LQR
    """

    def __init__(self, params: TractorTrailerParams):
        self.params = params
        self.mode = "approximate"
        self.design = "manual"
        # ゲイン K (3,): 3次システム [e1, e2, e3] に対するフィードバック
        self.K = np.array([1.0, 2.0, 3.0], dtype=float)
        self.v_ref = 1.0
        self.design_error: str | None = None
        # 設計パラメータの保存
        self._poles_raw: List = [
            {"re": -1, "im": 0}, {"re": -2, "im": 0}, {"re": -3, "im": 0},
        ]
        self._Q: List = [1, 1, 1]
        self._R: List = [1]

    def _get_system_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """現在のモードに応じた線形化モデル (A, B) を返す

        状態順: [y2, theta2, theta1]（近似）もしくは [z1, z2, z3]（厳密）
        """
        L1 = self.params.L1
        L2 = self.params.L2
        if self.mode == "exact":
            # 3重積分器: dz1 = z2, dz2 = z3, dz3 = u
            A = np.array([
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ])
            B = np.array([[0.0], [0.0], [1.0]])
        else:
            # 近似線形化 (theta1=theta2=0 周り):
            #   dy2/dt   = v * theta2
            #   dtheta2  = v * theta1 / L2
            #   dtheta1  = v * alpha / L1 - v * theta1 / L2
            # 時間微分を v で正規化（単位時間あたり進む距離=v として x 軸時間軸化）→
            #   A = [[0, 1, 0], [0, 0, 1/L2], [0, 0, -1/L2]], B = [[0], [0], [1/L1]]
            A = np.array([
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0 / L2],
                [0.0, 0.0, -1.0 / L2],
            ])
            B = np.array([[0.0], [0.0], [1.0 / L1]])
        return A, B

    def _redesign_gain(self) -> None:
        """保存された設計パラメータでゲインを再設計"""
        A, B = self._get_system_matrices()
        if self.design == "pole_assignment":
            self.K, self.design_error = _design_K_pole(A, B, self._poles_raw)
        elif self.design == "lqr":
            self.K, self.design_error = _design_K_lqr(A, B, self._Q, self._R)

    def set_control_params(self, control_params: Dict[str, Any]) -> None:
        """制御パラメータの更新"""
        if not isinstance(control_params, dict):
            return
        self.mode = control_params.get("type", self.mode)
        self.design = control_params.get("design", self.design)

        v_ref = control_params.get("v_ref")
        if v_ref is not None:
            self.v_ref = float(v_ref)

        if self.design == "pole_assignment":
            poles = control_params.get("poles")
            if poles is not None:
                self._poles_raw = poles
            self._redesign_gain()
        elif self.design == "lqr":
            Q = control_params.get("Q")
            R = control_params.get("R")
            if Q is not None:
                self._Q = Q
            if R is not None:
                self._R = R
            self._redesign_gain()
        else:
            # manual: 直接 K を指定
            K = control_params.get("K")
            if K is not None:
                try:
                    K_arr = np.asarray(K, dtype=float).flatten()
                    if K_arr.size == 3:
                        self.K = K_arr
                except Exception:
                    pass
            # k1, k2, k3 個別指定
            k1 = control_params.get("k1")
            k2 = control_params.get("k2")
            k3 = control_params.get("k3")
            if k1 is not None or k2 is not None or k3 is not None:
                self.K = np.array([
                    float(k1) if k1 is not None else self.K[0],
                    float(k2) if k2 is not None else self.K[1],
                    float(k3) if k3 is not None else self.K[2],
                ], dtype=float)

    def set_params(self, params: TractorTrailerParams) -> None:
        """物理パラメータ変更時にゲインを再設計"""
        self.params = params
        if self.design in ("pole_assignment", "lqr"):
            self._redesign_gain()

    def compute(self, state: TractorTrailerState, ref: RefSample) -> tuple[float, float]:
        """状態とリファレンスから制御入力 [v, alpha] を計算"""
        if ref is None:
            return 0.0, 0.0
        if self.mode == "exact":
            return self._exact_control(state, ref)
        else:
            return self._approximate_control(state, ref)

    def _approximate_control(self, state: TractorTrailerState, ref: RefSample) -> tuple[float, float]:
        """近似線形化による状態フィードバック制御
        誤差 e = [y2 - y2_ref, theta2 - theta2_ref, theta1 - theta1_ref]
        制御入力 alpha = -K @ e （theta1_ref = 0 を仮定）
        """
        v_ref = max(abs(ref.v), abs(self.v_ref)) if abs(ref.v) > 1e-6 else self.v_ref
        e = np.array([
            state.y2 - ref.pos[1],
            _wrap_angle(state.theta2 - ref.theta),
            _wrap_angle(state.theta1 - 0.0),
        ], dtype=float)

        alpha_cmd = -float(self.K @ e)
        # ±π/3 でクリップ（tan(alpha)発散防止）
        alpha_limit = math.pi / 3
        alpha_cmd = max(-alpha_limit, min(alpha_limit, alpha_cmd))
        return v_ref, alpha_cmd

    def _exact_control(self, state: TractorTrailerState, ref: RefSample) -> tuple[float, float]:
        """厳密線形化制御（x軸時間軸、3重積分器正準形）
        フラット出力 y2 に対して:
          z1 = y2
          z2 = dz1/dx2 = tan(theta2)
          z3 = dz2/dx2 = tan(theta1) / (L2 * cos(theta2))
        仮想入力 u = dz3/dx2, フィードバック u = -K @ [z1-z1r, z2-z2r, z3-z3r]
        alpha の復元は近似形を使用:
          tan(alpha) ≈ L1*cos(theta1)*cos(theta2) * (u*L2 + tan(theta1)^2*tan(theta2)/L2)
                      + L1 * tan(theta1) / L2
        """
        L1 = self.params.L1
        L2 = self.params.L2
        v_ref = max(abs(ref.v), abs(self.v_ref)) if abs(ref.v) > 1e-6 else self.v_ref

        if v_ref <= 1e-6:
            return 0.0, 0.0

        theta1 = state.theta1
        theta2 = state.theta2
        cos_t1 = math.cos(theta1)
        cos_t2 = math.cos(theta2)
        if abs(cos_t1) < 1e-6 or abs(cos_t2) < 1e-6:
            return v_ref, 0.0

        # 現在のフラット座標
        z1 = state.y2
        z2 = math.tan(theta2)
        z3 = math.tan(theta1) / (L2 * cos_t2)

        # 目標（ref.theta = theta2_ref、theta1_ref = 0 を仮定）
        cos_t2_ref = math.cos(ref.theta)
        if abs(cos_t2_ref) < 1e-6:
            return v_ref, 0.0
        z1_ref = ref.pos[1]
        z2_ref = math.tan(ref.theta)
        z3_ref = 0.0

        e = np.array([z1 - z1_ref, z2 - z2_ref, z3 - z3_ref], dtype=float)
        u = -float(self.K @ e)

        # alpha を仮想入力 u から復元（近似式）
        tan_t1 = math.tan(theta1)
        tan_t2 = math.tan(theta2)
        try:
            tan_alpha = (
                L1 * cos_t1 * cos_t2 * (u * L2 + tan_t1 * tan_t1 * tan_t2 / L2)
                + L1 * tan_t1 / L2
            )
            alpha_cmd = math.atan(tan_alpha)
        except Exception:
            alpha_cmd = 0.0

        alpha_cmd = max(-math.pi / 3, min(math.pi / 3, alpha_cmd))
        return float(v_ref), float(alpha_cmd)
