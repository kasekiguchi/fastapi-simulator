"""Linear observer (Luenberger) implementation.

Usage:
    from .base import ESTIMATOR
    est = ESTIMATOR(parameters=FurutaPendulumParams(), dt=0.01)
    est.set_estimator_params({"type": "observer", "design": {"type": "lqr", "Q": [...], "R": [...]}})

Inheritance:
    - Inherits _LinearEstimatorStrategy (defined in base.py).
    - Required methods to implement when extending: __init__, estimate.
    - Optional overrides: reset (if stateful), set_params (refresh gains).

Observer-specific API:
    - Supports design types mirroring controller FPControlParams:
        * "pole_assignment": poles -> place_poles on (Ad^T, Cd^T)
        * "state_feedback": uses provided gain vector as columns in Fo
        * "lqr": uses dlqr/lqr dual design on transpose system
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
from scipy import signal

from .base import (
    _LinearEstimatorStrategy,
    _state_vector,
    _to_square_matrix,
)
from ...utils.linear_optimal import lqr, dlqr
from ..common.physical_parameters import FurutaPendulumParams


class ObserverEstimator(_LinearEstimatorStrategy):
    """Luenberger observer based on the linearized model."""

    def __init__(self, params: FurutaPendulumParams, dt: float, design: Dict[str, Any], time_mode: str):
        """
        Args:
            params: Plant parameters used to linearize the model.
            dt: Simulation/control step (sec).
            design: FPControlParams-like dict to design observer gain Fo.
        """
        self.design = design or {}
        super().__init__(params, dt, time_mode)
        self.Fo = np.zeros((self.nx, self.ny))
        self._design_gain()

    def _design_gain(self) -> None:
        """Compute observer gain Fo based on selected design method."""
        design_type = self.design.get("type")
        if design_type == "pole_assignment":
            poles = self.design.get("poles") or self.design.get("pole") or []
            pole_list = self._parse_poles(poles)
            if not pole_list or self.Ad is None or self.Bd is None or self.Cd is None:
                self.Fo = np.zeros_like(self.Fo)
                return
            try:
                result = signal.place_poles(self.Ad.T, self.Cd.T, pole_list)
                self.Fo = np.asarray(result.gain_matrix, dtype=float).T
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
            Q = _to_square_matrix(self.design.get("Q") or [], self.nx)
            R = _to_square_matrix(self.design.get("R") or [], self.ny)
            if self.time_mode == "discrete" and self.Ad is not None and self.Bd is not None and self.Cd is not None:
                try:
                    self.Fo = dlqr(self.Ad.T, self.Cd.T, Q, R).T
                    return
                except Exception:
                    pass
            try:
                self.Fo = lqr(self.A.T, self.C.T, Q, R).T
            except Exception:
                self.Fo = np.zeros_like(self.Fo)
        else:
            self.Fo = np.zeros_like(self.Fo)

    @staticmethod
    def _parse_poles(poles: List[Dict[str, float]]):
        """Parse incoming pole descriptions into complex numbers."""
        parsed = []
        for p in poles:
            if isinstance(p, dict):
                re = float(p.get("re", 0.0))
                im = float(p.get("im", 0.0))
                parsed.append(complex(re, im))
                continue
            try:
                parsed.append(complex(p))
            except Exception:
                continue
        return parsed

    def set_params(self, params: FurutaPendulumParams) -> None:
        """Rebuild linear model and redesign gain when plant params change."""
        super().set_params(params)
        self.Fo = np.zeros((self.nx, self.ny))
        self._design_gain()

    def estimate(self, u: float, y) -> FurutaPendulumState:
        """Update observer state using y (shape 2) and input u."""
        y_vec = _state_vector(y, expected_dim=self.ny)
        xh = _state_vector(self.xh, expected_dim=self.nx)
        if self.time_mode == "discrete" and self.Ad is not None and self.Bd is not None and self.Cd is not None:
            # Discrete update: x_{k+1} = Ad x + Bd u + Fo (y - Cd x)
            xh = self.Ad @ xh + self.Bd.flatten() * float(u) + self.Fo @ (y_vec - self.Cd @ xh)
        else:
            # Continuous fallback (Euler step)
            dxh = self.A @ xh + self.B.flatten() * float(u) + self.Fo @ (y_vec - self.C @ xh)
            xh = xh + self.dt * dxh
        self.xh = xh
        return self.state.set(self.xh)
