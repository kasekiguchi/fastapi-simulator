"""LQR controller implementation.

Usage:
    from .base import CONTROLLER
    ctrl = CONTROLLER(parameters=FurutaPendulumParams(), dt=0.01)
    ctrl.set_control_params({"type": "lqr", "Q": [...], "R": [...]})

Inheritance:
    - Inherits _LinearControllerStrategy (defined in base.py).
    - Required methods to implement when extending: __init__, compute.
    - Optional overrides: reset (if stateful), set_params (refreshes gains).

Controller-specific API:
    - _design_gain(): designs K using dlqr (discrete) then lqr (continuous) fallback.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import (
    _LinearControllerStrategy,
    _state_vector,
    _to_square_matrix,
)
from ...utils.linear_optimal import lqr, dlqr
from ..common.physical_parameters import FurutaPendulumParams


class LQRController(_LinearControllerStrategy):
    """LQR gain computation with discrete/continuous selection."""

    def __init__(self, params: FurutaPendulumParams, dt: float, q: Sequence[float], r: Sequence[float], time_mode: str):
        """
        Args:
            params: Plant parameters used to linearize the model.
            dt: Simulation/control step (sec).
            q: Weighting matrix (flat list or diagonal entries) for states.
            r: Weighting matrix (flat list or diagonal entries) for input.
        """
        self.q = q
        self.r = r
        super().__init__(params, dt, time_mode)
        size_x = self.Ad.shape[0] if self.Ad is not None else self.nx
        size_u = self.Bd.shape[1] if self.Bd is not None else self.nu
        self.Q = _to_square_matrix(q, size_x)
        self.R = _to_square_matrix(r, size_u)
        self.K = np.zeros((size_u, size_x))
        self._design_gain()

    def _design_gain(self) -> None:
        """Compute LQR gain (dlqr preferred; lqr as fallback)."""
        if self.time_mode == "discrete" and self.Ad is not None and self.Bd is not None:
            try:
                self.K = dlqr(self.Ad, self.Bd, self.Q, self.R)
                return
            except Exception:
                pass
        try:
            self.K = lqr(self.A, self.B, self.Q, self.R)
        except Exception:
            self.K = np.zeros_like(self.K)

    def set_params(self, params: FurutaPendulumParams) -> None:
        """Rebuild model and redesign gain when plant params change."""
        super().set_params(params)
        size_x = self.Ad.shape[0] if self.Ad is not None else self.nx
        size_u = self.Bd.shape[1] if self.Bd is not None else self.nu
        self.Q = _to_square_matrix(self.q, size_x)
        self.R = _to_square_matrix(self.r, size_u)
        self.K = np.zeros((size_u, size_x))
        self._design_gain()

    def compute(self, state) -> float:
        """Return control torque u = -K x."""
        x = _state_vector(state, expected_dim=self.K.shape[1])
        try:
            return -float(self.K @ x[: self.K.shape[1]])
        except Exception:
            return 0.0
