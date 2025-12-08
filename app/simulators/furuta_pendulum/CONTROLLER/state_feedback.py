"""Simple state-feedback controller.

Usage:
    from .base import CONTROLLER
    ctrl = CONTROLLER(parameters=FurutaPendulumParams(), dt=0.01)
    ctrl.set_control_params({"type": "state_feedback", "gain": [k1, k2, k3, k4]})

Inheritance:
    - Inherits _BaseControllerStrategy (see base.py).
    - Required methods to implement when extending: __init__, compute.
    - Optional overrides: reset (if stateful), set_params (unused here).

Controller-specific API:
    - No extra public methods; compute() uses the provided gain vector directly.
"""

from __future__ import annotations

import numpy as np

from .base import _BaseControllerStrategy, _state_vector
from ..common.physical_parameters import FurutaPendulumParams


class StateFeedbackController(_BaseControllerStrategy):
    """Pure gain vector u = -Kx without model redesign."""

    def __init__(self, gain):
        """
        Args:
            gain: Iterable of gains [k1, k2, ...] applied to the state vector.
        """
        self.gain = np.asarray(gain, dtype=float).flatten()

    def set_params(self, params: FurutaPendulumParams) -> None:
        """No-op; this controller is plant-agnostic."""
        return

    def compute(self, state) -> float:
        """Return control torque u = -K x."""
        x = _state_vector(state, expected_dim=self.gain.size or 1)
        try:
            return -float(np.dot(self.gain, x[: self.gain.size]))
        except Exception:
            return 0.0
