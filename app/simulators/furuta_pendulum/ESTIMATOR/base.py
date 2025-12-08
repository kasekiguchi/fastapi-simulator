from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any

import numpy as np
from scipy import signal

from ..common.Ac import Ac_furutaPendulum
from ..common.Bc import Bc_furutaPendulum
from ..common.physical_parameters import FurutaPendulumParams
from ..common.state import FurutaPendulumState


@dataclass
class EstimatorParams:
    """Minimal estimator settings."""
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
    """Return continuous/discrete linearized models (A, B, C, Ad, Bd, Cd)."""
    A = Ac_furutaPendulum(params.as_array)
    B = Bc_furutaPendulum(params.as_array)
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    if time_mode == "discrete":
        try:
            Ad, Bd, Cd, _, _ = signal.cont2discrete((A, B, C, np.zeros((2, B.shape[1]))), dt)
        except Exception:
            Ad, Bd, Cd = None, None, None
    else:
        Ad, Bd, Cd = None, None, None
    return A, B, C, Ad, Bd, Cd


class _BaseEstimatorStrategy:
    """Abstract estimator contract used by ESTIMATOR."""

    def estimate(self, u: float, y) -> FurutaPendulumState:
        """Return updated state estimate from input u and measurement y."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal estimate/covariance."""
        pass

    def set_params(self, params: FurutaPendulumParams) -> None:
        """Update plant parameters if the estimator depends on them."""
        pass


class _LinearEstimatorStrategy(_BaseEstimatorStrategy):
    """Base for estimators that rely on the linearized Furuta model."""

    def __init__(self, params: FurutaPendulumParams, dt: float, time_mode: str):
        self.params = params
        self.dt = dt
        self.time_mode = time_mode
        self.state = FurutaPendulumState()
        self.xh = self.state.as_array
        self._rebuild_model()

    def _rebuild_model(self) -> None:
        """Recompute linearized A/B/C (continuous & discrete)."""
        self.A, self.B, self.C, self.Ad, self.Bd, self.Cd = _build_linear_model(self.params, self.dt, self.time_mode)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1] if self.B is not None and self.B.ndim > 1 else 1
        self.ny = self.C.shape[0]

    def set_params(self, params: FurutaPendulumParams) -> None:
        """Refresh internal model when plant params change."""
        self.params = params
        self._rebuild_model()

    def reset(self) -> None:
        self.state = FurutaPendulumState()
        self.xh = self.state.as_array


class ESTIMATOR:
    """
    Entry point to use Furuta Pendulum estimators from the simulator.
    """

    def __init__(
        self,
        parameters: Optional[FurutaPendulumParams] = None,
        settings: Optional[EstimatorParams] = None,
        dt: Optional[float] = None,
    ) -> None:
        self.params = parameters or FurutaPendulumParams()
        self.settings = settings or EstimatorParams()
        self.dt = float(dt if dt is not None else self.settings.dt)
        self.time_mode = self.settings.time_mode
        self.strategy: Optional[_BaseEstimatorStrategy] = None

    def estimate(self, u: float, y) -> FurutaPendulumState:
        if not self.strategy:
            return FurutaPendulumState()
        try:
            return self.strategy.estimate(u, y)
        except Exception:
            return FurutaPendulumState()

    def reset(self) -> None:
        if self.strategy:
            self.strategy.reset()

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))
        if self.strategy:
            self.strategy.set_params(self.params)

    def set_estimator_params(self, estimator_params: Optional[Dict[str, Any]] = None) -> None:
        """Select estimator type based on incoming payload."""
        if not estimator_params or "type" not in estimator_params:
            self.strategy = None
            return

        etype = estimator_params.get("type")
        if etype == "observer":
            from .observer import ObserverEstimator

            design = estimator_params.get("design") or {}
            self.strategy = ObserverEstimator(self.params, self.dt, design, self.time_mode)
        elif etype == "ekf":
            from .ekf import EKFEstimator

            sys_noise = estimator_params.get("sysNoise") or estimator_params.get("sys_noise") or []
            meas_noise = estimator_params.get("measNoise") or estimator_params.get("measure_noise") or []
            self.strategy = EKFEstimator(self.params, self.dt, sys_noise, meas_noise, self.time_mode)
        else:
            self.strategy = None
