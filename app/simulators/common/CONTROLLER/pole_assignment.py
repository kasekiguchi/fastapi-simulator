from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import signal

from .base import _LinearControllerStrategy
from ..linear_utils import state_vector


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
            print(f"[PoleAssignment] design with Ad/Bd: {self.Ad}, {self.Bd}")
            try:
                res = signal.place_poles(self.Ad, self.Bd, pole_list)
                print(f"[PoleAssignment] discrete gain matrix: {res.gain_matrix}")
                self.K = np.asarray(res.gain_matrix, dtype=float)
                return
            except Exception as e:
                self.design_error = f"{e}"
                print(f"[PoleAssignment] discrete design error (YT): {e}")
                # Try alternative algorithm in case default fails for near-defective cases
                try:
                    res_alt = signal.place_poles(self.Ad, self.Bd, pole_list, method="KNV0")
                    print(f"[PoleAssignment] discrete gain matrix (KNV0): {res_alt.gain_matrix}")
                    self.K = np.asarray(res_alt.gain_matrix, dtype=float)
                    self.design_error = None
                    return
                except Exception as e_alt:
                    self.design_error = f"{e_alt}"
                    print(f"[PoleAssignment] discrete design error (KNV0): {e_alt}")
        try:
            res = signal.place_poles(self.A, self.B, pole_list)
            print(f"[PoleAssignment] continuous gain matrix: {res.gain_matrix}")
            self.K = np.asarray(res.gain_matrix, dtype=float)
        except Exception as e2:
            self.K = np.zeros((self.nu, self.nx))
            self.design_error = f"{e2}"
            print(f"[PoleAssignment] continuous design error: {e2}")

    def compute(self, state):
        x = state_vector(state, expected_dim=self.nx)
        try:
            return -float(self.K @ x[: self.nx])
        except Exception:
            return 0.0
