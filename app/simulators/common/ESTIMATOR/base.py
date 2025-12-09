from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple

import numpy as np

from ..linear_utils import build_linear_model


@dataclass
class EstimatorParams:
    time_mode: str = "discrete"
    dt: float = 0.01


class GenericEstimator:
    """
    Generic linear estimator wrapper.
    matrices_fn: callable(params) -> (A, B, C)
    """

    def __init__(
        self,
        params,
        matrices_fn: Callable[[Any], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        settings: Optional[EstimatorParams] = None,
        dt: Optional[float] = None,
    ):
        self.params = params
        self.matrices_fn = matrices_fn
        self.settings = settings or EstimatorParams()
        self.dt = float(dt if dt is not None else self.settings.dt)
        self.time_mode = self.settings.time_mode
        self.strategy = None
        self.passthrough: bool = False

    def estimate(self, u: float, y):
        if not self.strategy:
            return None
        try:
            return self.strategy.estimate(u, y)
        except Exception:
            return None

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
        if not estimator_params or "type" not in estimator_params:
            self.strategy = None
            self.passthrough = False
            return
        tm = estimator_params.get("time_mode") or estimator_params.get("timeMode") or self.time_mode
        self.time_mode = tm
        etype = estimator_params.get("type")
        if etype == "none":
            # passthrough mode: use true state
            self.strategy = None
            self.passthrough = True
            return
        if etype == "observer":
            from .observer import ObserverEstimator

            design = estimator_params.get("design") or {}
            self.strategy = ObserverEstimator(self.params, self.dt, tm, self.matrices_fn, design)
        elif etype == "ekf":
            from .ekf import EKFEstimator

            sys_noise = estimator_params.get("sysNoise") or []
            meas_noise = estimator_params.get("measNoise") or []
            self.strategy = EKFEstimator(self.params, self.dt, tm, self.matrices_fn, sys_noise, meas_noise)
        else:
            self.strategy = None
            self.passthrough = False


class _BaseEstimatorStrategy:
    def estimate(self, u: float, y):
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def set_params(self, params) -> None:
        pass


class _LinearEstimatorStrategy(_BaseEstimatorStrategy):
    def __init__(self, params, dt: float, time_mode: str, matrices_fn: Callable[[Any], Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self.params = params
        self.dt = dt
        self.time_mode = time_mode
        self.matrices_fn = matrices_fn
        self.state = None
        self.xh = None
        self._rebuild_model()

    def _rebuild_model(self) -> None:
        self.A, self.B, self.C, self.Ad, self.Bd, self.Cd = build_linear_model(
            lambda: self.matrices_fn(self.params),
            self.dt,
            self.time_mode,
            include_output=True,
        )
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        if self.xh is None:
            self.xh = np.zeros(self.nx)

    def set_params(self, params) -> None:
        self.params = params
        self._rebuild_model()

    def reset(self) -> None:
        self.xh = None
