from __future__ import annotations

import numpy as np

from .base import _LinearEstimatorStrategy
from ..linear_utils import state_vector, to_square_matrix


class EKFEstimator(_LinearEstimatorStrategy):
    def __init__(self, params, dt: float, time_mode: str, matrices_fn, sys_noise, meas_noise):
        super().__init__(params, dt, time_mode, matrices_fn)
        self.P = np.eye(self.nx) * 1e-3
        self.Qw = to_square_matrix(sys_noise, self.nx)
        self.Rv = to_square_matrix(meas_noise, self.ny)

    def estimate(self, u: float, y):
        y_vec = state_vector(y, expected_dim=self.ny)
        if self.xh is None:
            try:
                C_use = self.Cd if (self.time_mode == "discrete" and self.Cd is not None) else self.C
                xh_init, *_ = np.linalg.lstsq(C_use, y_vec, rcond=None)
                self.xh = xh_init
            except Exception:
                self.xh = np.zeros(self.nx)
        xh = state_vector(self.xh, expected_dim=self.nx)
        # Unknown inputs (e.g., click) are not modeled in the estimator
        if self.time_mode == "discrete" and self.Ad is not None and self.Bd is not None and self.Cd is not None:
            xh_pred = self.Ad @ xh
            P_pred = self.Ad @ self.P @ self.Ad.T + self.Qw
            S = self.Cd @ P_pred @ self.Cd.T + self.Rv
            try:
                K = P_pred @ self.Cd.T @ np.linalg.inv(S)
            except Exception:
                K = np.zeros((self.nx, self.ny))
            xh = xh_pred + K @ (y_vec - self.Cd @ xh_pred)
            self.P = (np.eye(self.nx) - K @ self.Cd) @ P_pred
        else:
            dxh = self.A @ xh
            self.P = self.P + self.dt * (self.A @ self.P + self.P @ self.A.T + self.Qw)
            try:
                K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Rv)
            except Exception:
                K = np.zeros((self.nx, self.ny))
            xh = xh + self.dt * dxh + K @ (y_vec - self.C @ xh)
            self.P = (np.eye(self.nx) - K @ self.C) @ self.P
        self.xh = xh
        return xh
