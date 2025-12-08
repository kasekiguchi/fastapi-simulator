from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any

import numpy as np
from scipy import signal

from ..common.Ac import Ac_furutaPendulum
from ..common.Bc import Bc_furutaPendulum
from ..common.physical_parameters import FurutaPendulumParams


@dataclass
class ControllerParams:
    """Minimal controller settings."""
    time_mode: str = "discrete"
    dt: float = 0.01


def _state_vector(state, expected_dim: int = 4) -> np.ndarray:
    """Convert various state representations to a flat numpy vector."""
    try:
        if hasattr(state, "as_array"):
            arr = np.asarray(state.as_array, dtype=float).reshape(-1)
        else:
            arr = np.asarray(state, dtype=float).reshape(-1)
    except Exception:
        arr = np.zeros(expected_dim, dtype=float)

    if arr.size < expected_dim:
        arr = np.pad(arr, (0, expected_dim - arr.size))
    return arr


def _to_square_matrix(values: Sequence[float], size: int) -> np.ndarray:
    """Create a square matrix from a flat list. Falls back to identity when invalid."""
    arr = np.asarray(values, dtype=float).flatten()
    if arr.size == size * size:
        return arr.reshape((size, size))
    if arr.size >= size:
        return np.diag(arr[:size])
    if arr.size == 1:
        return np.diag(np.repeat(arr[0], size))
    return np.eye(size)


def _build_linear_model(params: FurutaPendulumParams, dt: float, time_mode: str):
    """Return continuous/discrete linearized models (A, B, Ad, Bd)."""
    A = Ac_furutaPendulum(params.as_array)
    B = Bc_furutaPendulum(params.as_array)
    if time_mode == "discrete":
        try:
            Ad, Bd, _, _, _ = signal.cont2discrete(
                (A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1]))),
                dt,
            )
        except Exception:
            # Fallback to continuous matrices to keep controllability for design
            Ad, Bd = A, B
    else:
        Ad, Bd = None, None
    return A, B, Ad, Bd


class _BaseControllerStrategy:
    """Abstract controller contract used by CONTROLLER."""

    def compute(self, state):
        """Return control input based on current state (array or object with as_array)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state (default: no-op)."""
        pass

    def set_params(self, params: FurutaPendulumParams) -> None:
        """Update plant parameters if the controller depends on them."""
        pass


class _LinearControllerStrategy(_BaseControllerStrategy):
    """Base for controllers that rely on the linearized Furuta model."""

    def __init__(self, params: FurutaPendulumParams, dt: float, time_mode: str):
        self.params = params
        self.dt = dt
        self.time_mode = time_mode
        self._rebuild_model()

    def _rebuild_model(self) -> None:
        """Recompute linearized A/B (continuous & discrete)."""
        self.A, self.B, self.Ad, self.Bd = _build_linear_model(self.params, self.dt, self.time_mode)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1] if self.B is not None and self.B.ndim > 1 else 1

    def set_params(self, params: FurutaPendulumParams) -> None:
        """Refresh internal model when plant params change."""
        self.params = params
        self._rebuild_model()


class CONTROLLER:
    """
    Entry point to use Furuta Pendulum controllers from the simulator.
    """

    def __init__(
        self,
        parameters: Optional[FurutaPendulumParams] = None,
        settings: Optional[ControllerParams] = None,
        dt: Optional[float] = None,
    ) -> None:
        self.params = parameters or FurutaPendulumParams()
        self.settings = settings or ControllerParams()
        self.dt = float(dt if dt is not None else self.settings.dt)
        self.time_mode = self.settings.time_mode
        self.strategy: Optional[_BaseControllerStrategy] = None

    def calc_input(self, xh) -> float:
        """Alias for MATLAB parity."""
        return self.compute(xh)

    def compute(self, state) -> float:
        if not self.strategy:
            return 0.0
        try:
            return float(self.strategy.compute(state))
        except Exception:
            return 0.0

    def reset(self) -> None:
        if self.strategy:
            self.strategy.reset()

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))
        if self.strategy:
            self.strategy.set_params(self.params)

    def set_control_params(self, control_params: Optional[Dict[str, Any]] = None) -> None:
        """Select controller type based on incoming payload."""
        if not control_params or "type" not in control_params:
            self.strategy = None
            return

        ctype = control_params.get("type")
        tm = control_params.get("time_mode") or control_params.get("timeMode") or self.time_mode
        self.time_mode = tm
        if ctype == "pole_assignment":
            from .pole_assignment import PoleAssignmentController

            poles = control_params.get("poles") or control_params.get("pole") or []
            self.strategy = PoleAssignmentController(self.params, self.dt, poles, tm)
        elif ctype == "state_feedback":
            from .state_feedback import StateFeedbackController

            gain = control_params.get("gain") or []
            self.strategy = StateFeedbackController(gain)
        elif ctype == "lqr":
            from .lqr_controller import LQRController

            q = control_params.get("Q") or []
            r = control_params.get("R") or []
            self.strategy = LQRController(self.params, self.dt, q, r, tm)
        else:
            self.strategy = None
