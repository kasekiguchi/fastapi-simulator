"""Pole-placement controller implementation.

Usage:
    from .base import CONTROLLER
    ctrl = CONTROLLER(parameters=FurutaPendulumParams(), dt=0.01)
    ctrl.set_control_params({"type": "pole_assignment", "poles": [{"re": -2, "im": 1}]})

Inheritance:
    - Inherits _LinearControllerStrategy (defined in base.py).
    - Required methods to implement when extending: __init__, compute.
    - Optional overrides: reset (if stateful), set_params (if model-dependent).

Controller-specific API:
    - _design_gain(): recompute feedback gain K when poles or parameters change.
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np
from scipy import signal

from .base import (
    _LinearControllerStrategy,
    _state_vector,
)
from ..common.physical_parameters import FurutaPendulumParams


class PoleAssignmentController(_LinearControllerStrategy):
    """State-feedback via pole placement for the discrete linear model."""

    def __init__(self, params: FurutaPendulumParams, dt: float, poles: List[Dict[str, float]], time_mode: str):
        """
        Args:
            params: Plant parameters used to linearize the model.
            dt: Simulation/control step (sec).
            poles: Target eigenvalues as dicts {"re": float, "im": float} or complex-like values.
        """
        self.poles_raw = poles
        super().__init__(params, dt, time_mode)
        self.K = np.zeros((self.nu, self.nx))
        self.design_error: str | None = None
        self._design_gain()

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

    def _normalize_poles(self, poles):
        """Ensure conjugate pairs exist and truncate/extend to state dimension."""
        poles = list(poles)
        # Add missing conjugates for non-zero imaginary parts
        for p in list(poles):
            if abs(p.imag) > 0 and np.conj(p) not in poles:
                poles.append(np.conj(p))
        # Trim or pad with small negative real poles to match nx
        if len(poles) < self.nx:
            poles.extend([complex(-1.0 * (i + 1), 0) for i in range(self.nx - len(poles))])
        return poles[: self.nx]

    def _design_gain(self) -> None:
        """Compute gain matrix K for the current model and pole targets."""
        self.design_error = None
        pole_list = self._normalize_poles(self._parse_poles(self.poles_raw))
        if not pole_list:
            self.K = np.zeros_like(self.K)
            self.design_error = "no poles provided"
            return
        # Choose model based on time_mode
        if self.time_mode == "discrete" and self.Ad is not None and self.Bd is not None:
            try:
                result = signal.place_poles(self.Ad, self.Bd, pole_list)
                self.K = np.asarray(result.gain_matrix, dtype=float)
                return
            except Exception as e:
                self.design_error = f"{e}"
        # Continuous fallback
        try:
            result = signal.place_poles(self.A, self.B, pole_list)
            self.K = np.asarray(result.gain_matrix, dtype=float)
        except Exception as e2:
            self.K = np.zeros_like(self.K)
            self.design_error = f"{e2}"

    def set_params(self, params: FurutaPendulumParams) -> None:
        """Rebuild linear model and redesign gain when plant params change."""
        super().set_params(params)
        self._design_gain()

    def compute(self, state) -> float:
        """Return control torque u = -K x."""
        x = _state_vector(state, expected_dim=self.K.shape[1])
        try:
            return -float(self.K @ x[: self.K.shape[1]])
        except Exception:
            return 0.0
