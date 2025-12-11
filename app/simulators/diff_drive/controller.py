from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np
from scipy import linalg

from .reference import RefSample
from .state import DiffDriveState, DiffDriveParams


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class DiffDriveController:
    def __init__(self, params: DiffDriveParams):
        self.params = params
        self.mode = "pid"
        self.gains = {"kx": 1.0, "ky": 2.0, "ktheta": 1.5}
        self.q = np.diag([2.0, 4.0, 2.0])
        self.r = np.diag([1.0, 1.0])

    def set_control_params(self, control_params: Dict[str, Any]) -> None:
        if not isinstance(control_params, dict):
            return
        self.mode = control_params.get("type", self.mode)
        kx = control_params.get("kx")
        ky = control_params.get("ky")
        ktheta = control_params.get("ktheta")
        if kx is not None:
            self.gains["kx"] = float(kx)
        if ky is not None:
            self.gains["ky"] = float(ky)
        if ktheta is not None:
            self.gains["ktheta"] = float(ktheta)
        q = control_params.get("Q") or control_params.get("q")
        r = control_params.get("R") or control_params.get("r")
        if q:
            try:
                q_arr = np.asarray(q, dtype=float).flatten()
                if q_arr.size >= 3:
                    self.q = np.diag(q_arr[:3])
            except Exception:
                pass
        if r:
            try:
                r_arr = np.asarray(r, dtype=float).flatten()
                if r_arr.size >= 2:
                    self.r = np.diag(r_arr[:2])
            except Exception:
                pass

    def compute(self, state: DiffDriveState, ref: RefSample) -> tuple[float, float]:
        if ref is None:
            return 0.0, 0.0
        dx = ref.pos[0] - state.x
        dy = ref.pos[1] - state.y
        e_theta = _wrap_angle(ref.theta - state.theta)
        c, s = math.cos(ref.theta), math.sin(ref.theta)
        e_x = c * dx + s * dy
        e_y = -s * dx + c * dy
        if self.mode == "lqr":
            v_cmd, omega_cmd = self._lqr_control(ref, e_x, e_y, e_theta)
        else:
            v_cmd = ref.v + self.gains["kx"] * e_x
            omega_cmd = ref.omega + self.gains["ky"] * e_y + self.gains["ktheta"] * e_theta
        return float(v_cmd), float(omega_cmd)

    def _lqr_control(self, ref: RefSample, e_x: float, e_y: float, e_theta: float) -> tuple[float, float]:
        v_ref = max(abs(ref.v), 1e-3)
        omega_ref = ref.omega
        A = np.array(
            [
                [0.0, omega_ref, 0.0],
                [-omega_ref, 0.0, v_ref],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        B = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=float)
        try:
            P = linalg.solve_continuous_are(A, B, self.q, self.r)
            K = np.linalg.inv(self.r) @ (B.T @ P)
            e = np.array([e_x, e_y, e_theta], dtype=float)
            u_fb = -K @ e
            v = ref.v + u_fb[0]
            omega = ref.omega + u_fb[1]
            return v, omega
        except Exception:
            v = ref.v + self.gains["kx"] * e_x
            omega = ref.omega + self.gains["ky"] * e_y + self.gains["ktheta"] * e_theta
            return v, omega
