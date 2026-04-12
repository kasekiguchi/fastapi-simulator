from __future__ import annotations

import math
from typing import Dict, Any, List

import numpy as np
from scipy import signal, linalg
from ..common.linear_utils import ackermann

from .reference import RefSample
from .state import TricycleState, TricycleParams


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
    # 極の数を合わせる
    if len(poles) < nx:
        poles.extend([complex(-1.0 * (i + 1), 0) for i in range(nx - len(poles))])
    poles = poles[:nx]
    # 共役対の補完
    for p in list(poles):
        if abs(p.imag) > 0 and np.conj(p) not in poles:
            poles.append(np.conj(p))
    poles = poles[:nx]
    # 1. place_poles (YT法)
    try:
        res = signal.place_poles(A, B, poles)
        K = np.asarray(res.gain_matrix, dtype=float).flatten()
        print(f"[Tricycle] pole placement (YT): poles={poles} -> K={K}", flush=True)
        return K, None
    except Exception as e:
        print(f"[Tricycle] YT failed: {e}", flush=True)
    # 2. Ackermann法（単入力系、重極対応）
    try:
        K = ackermann(A, B, poles).flatten()
        print(f"[Tricycle] pole placement (Ackermann): poles={poles} -> K={K}", flush=True)
        return K, None
    except Exception as e:
        print(f"[Tricycle] Ackermann failed: {e}", flush=True)
        return np.zeros(nx, dtype=float), str(e)


def _design_K_lqr(A: np.ndarray, B: np.ndarray, Q_vals: List, R_vals: List) -> tuple[np.ndarray, str | None]:
    """LQR でゲイン K を設計（連続系）"""
    nx = A.shape[0]
    nu = B.shape[1]
    # Q, R 行列の構築
    Q_arr = np.asarray(Q_vals, dtype=float).flatten()
    R_arr = np.asarray(R_vals, dtype=float).flatten()
    Q = np.diag(Q_arr[:nx]) if Q_arr.size >= nx else np.eye(nx)
    R = np.diag(R_arr[:nu]) if R_arr.size >= nu else np.eye(nu)
    try:
        P = linalg.solve_continuous_are(A, B, Q, R)
        K = (np.linalg.inv(R) @ B.T @ P).flatten()
        print(f"[Tricycle] LQR: Q=diag({np.diag(Q)}), R=diag({np.diag(R)}) -> K={K}", flush=True)
        return K, None
    except Exception as e:
        print(f"[Tricycle] LQR CARE failed: {e}", flush=True)
        return np.zeros(nx, dtype=float), str(e)


class TricycleController:
    """三輪車モデルのコントローラ

    制御モード (type):
      - "approximate": 近似線形化による状態フィードバック
      - "exact": 厳密線形化（x軸時間軸状態制御正準形）

    ゲイン設計法 (design):
      - "manual": 直接 K を指定
      - "pole_assignment": 極配置法
      - "lqr": LQR
    """

    def __init__(self, params: TricycleParams):
        self.params = params
        self.mode = "approximate"
        self.design = "manual"
        # ゲイン K (2,): 2次システム [e1, e2] に対するフィードバック
        self.K = np.array([1.0, 2.0], dtype=float)
        self.v_ref = 1.0
        self.design_error: str | None = None
        # 設計パラメータの保存（パラメータ変更時の再設計用）
        self._poles_raw: List = [{"re": -1, "im": 0}, {"re": -2, "im": 0}]
        self._Q: List = [1, 1]
        self._R: List = [1]

    def _get_system_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """現在のモードに応じた線形化モデル (A, B) を返す"""
        L = self.params.L
        if self.mode == "exact":
            # 二重積分器: dz1/dx = z2, dz2/dx = u
            A = np.array([[0.0, 1.0], [0.0, 0.0]])
            B = np.array([[0.0], [1.0]])
        else:
            # 近似線形化 (theta=0 周り): dy/dx = theta, dtheta/dx = alpha/L
            A = np.array([[0.0, 1.0], [0.0, 0.0]])
            B = np.array([[0.0], [1.0 / L]])
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
                    if K_arr.size == 2:
                        self.K = K_arr
                except Exception:
                    pass
            # k1, k2 個別指定
            k1 = control_params.get("k1")
            k2 = control_params.get("k2")
            if k1 is not None or k2 is not None:
                self.K = np.array([
                    float(k1) if k1 is not None else self.K[0],
                    float(k2) if k2 is not None else self.K[1],
                ], dtype=float)

    def set_params(self, params: TricycleParams) -> None:
        """物理パラメータ変更時にゲインを再設計"""
        self.params = params
        if self.design in ("pole_assignment", "lqr"):
            self._redesign_gain()

    def compute(self, state: TricycleState, ref: RefSample) -> tuple[float, float]:
        """状態とリファレンスから制御入力 [v, alpha] を計算"""
        if ref is None:
            return 0.0, 0.0
        if self.mode == "exact":
            return self._exact_control(state, ref)
        else:
            return self._approximate_control(state, ref)

    def _approximate_control(self, state: TricycleState, ref: RefSample) -> tuple[float, float]:
        """近似線形化による状態フィードバック制御
        線形化モデル (theta=0 周り): A = [[0, 1], [0, 0]], B = [[0], [1/L]]
        誤差 e = [y_err, theta_err]
        制御入力 alpha = -K @ e
        """
        v_ref = max(abs(ref.v), abs(self.v_ref)) if abs(ref.v) > 1e-6 else self.v_ref
        e = np.array([
            state.y - ref.pos[1],
            _wrap_angle(state.theta - ref.theta),
        ], dtype=float)

        alpha_cmd = -float(self.K @ e)
        # ±π/3 でクリップ（π/2 だと tan(alpha) が発散してODEが破綻する）
        alpha_limit = math.pi / 3
        alpha_cmd = max(-alpha_limit, min(alpha_limit, alpha_cmd))
        return v_ref, alpha_cmd

    def _exact_control(self, state: TricycleState, ref: RefSample) -> tuple[float, float]:
        """厳密線形化制御（x軸時間軸状態制御正準形）
        z1 = y, z2 = tan(theta)
        仮想入力 u = -K @ [z1_err, z2_err]
        操舵角の復元: alpha = atan(u * L * cos^3(theta))
        """
        L = self.params.L
        v_ref = max(abs(ref.v), abs(self.v_ref)) if abs(ref.v) > 1e-6 else self.v_ref

        if v_ref <= 1e-6:
            return 0.0, 0.0

        theta = state.theta
        cos_theta = math.cos(theta)
        if abs(cos_theta) < 1e-6:
            return v_ref, 0.0

        z1 = state.y
        z2 = math.tan(theta)
        z1_ref = ref.pos[1]
        cos_theta_ref = math.cos(ref.theta)
        z2_ref = math.tan(ref.theta) if abs(cos_theta_ref) > 1e-6 else 0.0

        e = np.array([z1 - z1_ref, z2 - z2_ref], dtype=float)
        u_exact = -float(self.K @ e)

        cos3_theta = cos_theta ** 3
        alpha_cmd = math.atan(u_exact * L * cos3_theta)
        alpha_cmd = max(-math.pi / 3, min(math.pi / 3, alpha_cmd))

        return float(v_ref), float(alpha_cmd)
