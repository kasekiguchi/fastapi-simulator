from __future__ import annotations

import numpy as np
from scipy import signal

from .base import _LinearEstimatorStrategy
from ..linear_utils import state_vector, to_square_matrix


class ObserverEstimator(_LinearEstimatorStrategy):
    def __init__(self, params, dt: float, time_mode: str, matrices_fn, design):
        self.design = design or {}
        super().__init__(params, dt, time_mode, matrices_fn)
        self.Fo = np.zeros((self.nx, self.ny))
        self._design_gain()

    def _design_gain(self):
        design_type = self.design.get("type")
        if design_type == "pole_assignment":
            poles = self.design.get("poles") or []
            pole_list = []
            for p in poles:
                if isinstance(p, dict):
                    pole_list.append(complex(float(p.get("re", 0.0)), float(p.get("im", 0.0))))
                else:
                    try:
                        pole_list.append(complex(p))
                    except Exception:
                        continue
            if not pole_list:
                self.Fo = np.zeros_like(self.Fo)
                return
            if self.time_mode == "discrete" and self.Ad is not None and self.Cd is not None:
                try:
                    res = signal.place_poles(self.Ad.T, self.Cd.T, pole_list)
                    self.Fo = np.asarray(res.gain_matrix, dtype=float).T
                    return
                except Exception:
                    pass
            try:
                res = signal.place_poles(self.A.T, self.C.T, pole_list)
                self.Fo = np.asarray(res.gain_matrix, dtype=float).T
            except Exception:
                self.Fo = np.zeros_like(self.Fo)
        elif design_type == "state_feedback":
            gain = np.asarray(self.design.get("gain") or [], dtype=float).flatten()
            if gain.size == self.nx * self.ny:
                self.Fo = gain.reshape((self.nx, self.ny))
            elif gain.size >= self.nx:
                self.Fo = np.diag(gain[: self.nx])
            else:
                self.Fo = np.zeros_like(self.Fo)
        elif design_type == "lqr":
            Q = to_square_matrix(self.design.get("Q") or [], self.nx)
            R = to_square_matrix(self.design.get("R") or [], self.ny)
            if self.time_mode == "discrete" and self.Ad is not None and self.Cd is not None:
                try:
                    self.Fo = np.asarray(signal.dlti(self.Ad, self.Cd).dare(Q, R)[0], dtype=float).T  # type: ignore
                    return
                except Exception:
                    pass
            try:
                self.Fo = np.asarray(signal.lti(self.A, self.C).care(Q, R)[0], dtype=float).T  # type: ignore
            except Exception:
                self.Fo = np.zeros_like(self.Fo)
        else:
            self.Fo = np.zeros_like(self.Fo)

    def estimate(self, u: float, y):
        y_vec = state_vector(y, expected_dim=self.ny)
        if self.xh is None:
            # initialize with measured outputs (pseudo-inverse)
            try:
                C_use = self.Cd if (self.time_mode == "discrete" and self.Cd is not None) else self.C
                xh_init, *_ = np.linalg.lstsq(C_use, y_vec, rcond=None)
                self.xh = xh_init
            except Exception:
                self.xh = np.zeros(self.nx)
        xh = state_vector(self.xh, expected_dim=self.nx)
        # Unknown inputs (e.g., click) are ignored in the observer model
        if self.time_mode == "discrete" and self.Ad is not None and self.Cd is not None:
            xh = self.Ad @ xh + self.Fo @ (y_vec - self.Cd @ xh)
        else:
            dxh = self.A @ xh + self.Fo @ (y_vec - self.C @ xh)
            xh = xh + self.dt * dxh
        self.xh = xh
        return xh
