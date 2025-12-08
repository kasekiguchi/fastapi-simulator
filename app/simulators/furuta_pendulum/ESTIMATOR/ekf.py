"""Extended Kalman Filter (EKF) implementation (linearized-model version).

Usage:
    from .base import ESTIMATOR
    est = ESTIMATOR(parameters=FurutaPendulumParams(), dt=0.01)
    est.set_estimator_params({"type": "ekf", "sysNoise": [...], "measNoise": [...]})

Inheritance:
    - Inherits _LinearEstimatorStrategy (defined in base.py).
    - Required methods to implement when extending: __init__, estimate.
    - Optional overrides: reset (if stateful), set_params (refresh gains/linear model).

Estimator-specific API:
    - set_process_noise(Qw) / set_measure_noise(Rv) helpers to tune covariance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import _LinearEstimatorStrategy, _state_vector, _to_square_matrix
from ..common.physical_parameters import FurutaPendulumParams


class EKFEstimator(_LinearEstimatorStrategy):
    """EKF-style estimator using linearized A/B/C (acts as KF here)."""

    def __init__(
        self,
        params: FurutaPendulumParams,
        dt: float,
        sys_noise: Sequence[float],
        meas_noise: Sequence[float],
        time_mode: str,
    ):
        """
        Args:
            params: Plant parameters used to linearize the model.
            dt: Simulation/control step (sec).
            sys_noise: Process noise covariance entries.
            meas_noise: Measurement noise covariance entries.
        """
        super().__init__(params, dt, time_mode)
        self.P = np.eye(self.nx) * 1e-3  # covariance
        self.set_process_noise(sys_noise)
        self.set_measure_noise(meas_noise)

    def set_process_noise(self, q) -> None:
        """Update process noise covariance Qw."""
        self.Qw = _to_square_matrix(q, self.nx)

    def set_measure_noise(self, r) -> None:
        """Update measurement noise covariance Rv."""
        self.Rv = _to_square_matrix(r, self.ny)

    def set_params(self, params: FurutaPendulumParams) -> None:
        """Rebuild model when plant params change (keeps noise/covariance)."""
        super().set_params(params)
        self.P = np.eye(self.nx) * 1e-3

    def reset(self) -> None:
        super().reset()
        self.P = np.eye(self.nx) * 1e-3

    def estimate(self, u: float, y) -> FurutaPendulumState:
        """Kalman filter predict/update using linearized model."""
        y_vec = _state_vector(y, expected_dim=self.ny)
        xh = _state_vector(self.xh, expected_dim=self.nx)

        if self.Ad is not None and self.Bd is not None and self.Cd is not None:
            # Predict
            xh_pred = self.Ad @ xh + self.Bd.flatten() * float(u)
            P_pred = self.Ad @ self.P @ self.Ad.T + self.Qw
            # Update
            S = self.Cd @ P_pred @ self.Cd.T + self.Rv
            try:
                K = P_pred @ self.Cd.T @ np.linalg.inv(S)
            except Exception:
                K = np.zeros((self.nx, self.ny))
            xh = xh_pred + K @ (y_vec - self.Cd @ xh_pred)
            self.P = (np.eye(self.nx) - K @ self.Cd) @ P_pred
        else:
            # Continuous-time Euler KF update (approx)
            dxh = self.A @ xh + self.B.flatten() * float(u)
            self.P = self.P + self.dt * (self.A @ self.P + self.P @ self.A.T + self.Qw)
            try:
                K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Rv)
            except Exception:
                K = np.zeros((self.nx, self.ny))
            xh = xh + self.dt * dxh + K @ (y_vec - self.C @ xh)
            self.P = (np.eye(self.nx) - K @ self.C) @ self.P

        self.xh = xh
        return self.state.set(self.xh)
