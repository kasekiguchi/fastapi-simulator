from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import signal

from .base import _LinearControllerStrategy
from ..linear_utils import state_vector, ackermann


class PoleAssignmentController(_LinearControllerStrategy):
    def __init__(self, params, dt: float, time_mode: str, matrices_fn: Callable[[Any], tuple], poles):
        self.poles_raw = poles or []
        self.design_error = None
        super().__init__(params, dt, time_mode, matrices_fn)
        self.K = np.zeros((self.nu, self.nx))
        self._design_gain()

    def _parse_poles(self):
        parsed = []
        print(f"[PoleAssignment] raw poles: {self.poles_raw}")
        for p in self.poles_raw:
            if isinstance(p, dict):
                parsed.append(complex(float(p.get("re", 0.0)), float(p.get("im", 0.0))))
            else:
                try:
                    parsed.append(complex(p))
                except Exception:
                    continue
        print(f"[PoleAssignment] parsed poles: {parsed}")
        return parsed

    def _adapt_poles_to_time_mode(self, poles):
        """Convert provided poles to match the design domain.

        - If time_mode is continuous but all poles look like discrete (|p|<1),
          map them via log(p)/dt to continuous s-plane.
        - If time_mode is discrete but poles look continuous (Re(p)<0),
          map them via exp(p*dt) to the z-plane.
        """
        arr = np.array(poles, dtype=complex)
        if arr.size == 0:
            return poles

        if self.time_mode == "continuous":
            if np.all(np.abs(arr) < 1.0 + 1e-8):
                arr = np.array([np.log(p) / self.dt if np.abs(p) > 0 else -1.0 / self.dt for p in arr], dtype=complex)
        else:  # discrete
            if np.any(arr.real < 0):
                arr = np.array([np.exp(p * self.dt) for p in arr], dtype=complex)
        return list(arr)

    def _normalize_poles(self, poles):
        poles = list(poles)
        for p in list(poles):
            if abs(p.imag) > 0 and np.conj(p) not in poles:
                poles.append(np.conj(p))
        if len(poles) < self.nx:
            poles.extend([complex(-1.0 * (i + 1), 0) for i in range(self.nx - len(poles))])
        return poles[: self.nx]

    def _design_gain(self):
        self.design_error = None
        pole_list = self._normalize_poles(self._parse_poles())
        print(f"[PoleAssignment] normalized poles({self.time_mode}): {pole_list}")
        if not pole_list:
            self.K = np.zeros((self.nu, self.nx))
            self.design_error = "no poles provided"
            return
        if self.time_mode == "discrete" and self.Ad is not None and self.Bd is not None:
            A_use, B_use = self.Ad, self.Bd
        else:
            A_use, B_use = self.A, self.B
        # 1. place_poles (YT法)
        try:
            res = signal.place_poles(A_use, B_use, pole_list)
            self.K = np.asarray(res.gain_matrix, dtype=float)
            print(f"[PoleAssignment] {self.time_mode} gain (YT): {self.K}", flush=True)
            return
        except Exception as e:
            print(f"[PoleAssignment] YT failed: {e}", flush=True)
        # 2. place_poles (KNV0法)
        try:
            res = signal.place_poles(A_use, B_use, pole_list, method="KNV0")
            self.K = np.asarray(res.gain_matrix, dtype=float)
            self.design_error = None
            print(f"[PoleAssignment] {self.time_mode} gain (KNV0): {self.K}", flush=True)
            return
        except Exception as e:
            print(f"[PoleAssignment] KNV0 failed: {e}", flush=True)
        # 3. Ackermann法（単入力系、重極対応）
        if self.nu == 1:
            try:
                K_acker = ackermann(A_use, B_use, pole_list)
                self.K = K_acker.reshape(1, -1)
                self.design_error = None
                print(f"[PoleAssignment] {self.time_mode} gain (Ackermann): {self.K}", flush=True)
                return
            except Exception as e:
                print(f"[PoleAssignment] Ackermann failed: {e}", flush=True)
        self.K = np.zeros((self.nu, self.nx))
        self.design_error = "All pole placement methods failed"
        print(f"[PoleAssignment] all methods failed", flush=True)

    def compute(self, state):
        x = state_vector(state, expected_dim=self.nx)
        try:
            return -float((self.K @ x[: self.nx]).item())
        except Exception as e:
            print(f"[PoleAssignment] compute error: {e}, K.shape={self.K.shape}, x={x}", flush=True)
            return 0.0
