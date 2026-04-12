from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np

from .reference import RefSample
from .state import TricycleState, TricycleParams


def _wrap_angle(a: float) -> float:
    """角度を [-pi, pi) に正規化"""
    return (a + math.pi) % (2 * math.pi) - math.pi


class TricycleController:
    """三輪車モデルのコントローラ

    2つの制御モード:
      - "approximate": 近似線形化による状態フィードバック
      - "exact": 厳密線形化（x軸時間軸状態制御正準形）
    """

    def __init__(self, params: TricycleParams):
        self.params = params
        self.mode = "approximate"
        # ゲイン K (2,): 2次システム [e1, e2] に対するフィードバック
        # approximate: alpha = -K @ [y_err, theta_err]
        # exact: u_exact = -K @ [z1_err, z2_err]
        self.K = np.array([1.0, 2.0], dtype=float)
        self.v_ref = 1.0

    def set_control_params(self, control_params: Dict[str, Any]) -> None:
        """制御パラメータの更新"""
        if not isinstance(control_params, dict):
            return
        self.mode = control_params.get("type", self.mode)

        # 近似線形化用ゲイン
        K = control_params.get("K")
        if K is not None:
            try:
                K_arr = np.asarray(K, dtype=float).flatten()
                if K_arr.size == 2:
                    self.K = K_arr
            except Exception:
                pass

        v_ref = control_params.get("v_ref")
        if v_ref is not None:
            self.v_ref = float(v_ref)

        # k1, k2 が個別指定された場合も K に統合
        k1 = control_params.get("k1")
        k2 = control_params.get("k2")
        if k1 is not None or k2 is not None:
            self.K = np.array([
                float(k1) if k1 is not None else self.K[0],
                float(k2) if k2 is not None else self.K[1],
            ], dtype=float)

    def compute(self, state: TricycleState, ref: RefSample) -> tuple[float, float]:
        """状態とリファレンスから制御入力 [v, alpha] を計算"""
        if ref is None:
            return 0.0, 0.0
        if self.mode == "exact":
            return self._exact_control(state, ref)
        else:
            return self._approximate_control(state, ref)

    def _approximate_control(self, state: TricycleState, ref: RefSample) -> tuple[float, float]:
        """時間軸状態制御形の近似線形化による状態フィードバック制御
        時間軸状態制御形: z = [x, y, theta], u = [v, alpha]
            dx/dt = v * cos(theta)
            dy/dx = tan(theta)
            dtheta/dx = tan(alpha) / (L*cos(theta))
        線形化モデル: A = [[0, 1/cos^2(theta)], [0, 0]], B = [[0], [1/L]]
        誤差 e = [x - x_ref, y - y_ref, theta - theta_ref]
        制御入力 u = [v_ref, -K @ e[1:3]]
        """
        v_ref = max(abs(ref.v), abs(self.v_ref)) if abs(ref.v) > 1e-6 else self.v_ref
        # 誤差ベクトル [y_err, theta_err]
        e = np.array([
            state.y - ref.pos[1],
            _wrap_angle(state.theta - ref.theta),
        ], dtype=float)

        # フィードバック: alpha = -K @ [y_err, theta_err]
        u = np.array([v_ref, -float(self.K @ e)], dtype=float)

        v_cmd = float(u[0])
        alpha_cmd = float(u[1])

        # 操舵角の制限（±π/2 ≈ ±90度）
        alpha_cmd = max(-math.pi / 2, min(math.pi / 2, alpha_cmd))

        return v_cmd, alpha_cmd

    def _exact_control(self, state: TricycleState, ref: RefSample) -> tuple[float, float]:
        """厳密線形化制御（x軸時間軸状態制御正準形）

        独立変数をtからxに変換:
          z1 = y, z2 = tan(theta)
          dz1/dx = tan(theta) = z2
          dz2/dx = tan(alpha) / (L * cos^3(theta))
        仮想入力 u_exact = dz2/dx で二重積分器になる
        状態フィードバック: u_exact = -k1*(z1 - y_ref) - k2*(z2 - tan(theta_ref))
        操舵角の復元: alpha = atan(u_exact * L * cos^3(theta))
        """
        L = self.params.L
        v_ref = max(abs(ref.v), abs(self.v_ref)) if abs(ref.v) > 1e-6 else self.v_ref

        # v > 0 でないと厳密線形化は機能しない
        if v_ref <= 1e-6:
            print("[TricycleController] exact linearization requires v > 0, falling back to zero", flush=True)
            return 0.0, 0.0

        theta = state.theta
        cos_theta = math.cos(theta)

        # cos(theta) ≈ 0 のとき特異点 → フォールバック
        if abs(cos_theta) < 1e-6:
            print("[TricycleController] exact linearization singular (cos(theta)≈0), falling back", flush=True)
            return v_ref, 0.0

        # 状態変数
        z1 = state.y
        z2 = math.tan(theta)

        # 目標状態
        z1_ref = ref.pos[1]
        theta_ref = ref.theta
        cos_theta_ref = math.cos(theta_ref)
        z2_ref = math.tan(theta_ref) if abs(cos_theta_ref) > 1e-6 else 0.0

        # 仮想入力（二重積分器のフィードバック）: u = -K @ [z1_err, z2_err]
        e = np.array([z1 - z1_ref, z2 - z2_ref], dtype=float)
        u_exact = -float(self.K @ e)

        # 操舵角の復元
        cos3_theta = cos_theta ** 3
        alpha_cmd = math.atan(u_exact * L * cos3_theta)

        # 操舵角の制限
        alpha_cmd = max(-math.pi / 3, min(math.pi / 3, alpha_cmd))

        return float(v_ref), float(alpha_cmd)
