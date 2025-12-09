from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple

import numpy as np

from ..linear_utils import build_linear_model


@dataclass
class ControllerParams:
    time_mode: str = "discrete"  # "discrete" | "continuous"
    dt: float = 0.01


class GenericController:
    """
    Generic linear controller wrapper.
    matrices_fn: callable(params) -> (A, B)
    """

    def __init__(
        self,
        params,
        matrices_fn: Callable[[Any], Tuple[np.ndarray, np.ndarray]],
        settings: Optional[ControllerParams] = None,
        dt: Optional[float] = None,
    ):
        self.params = params
        self.matrices_fn = matrices_fn
        self.settings = settings or ControllerParams()
        self.dt = float(dt if dt is not None else self.settings.dt)
        self.time_mode = self.settings.time_mode
        self.strategy = None

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
        if not control_params or "type" not in control_params:
            self.strategy = None
            return
        tm = control_params.get("time_mode") or control_params.get("timeMode") or self.time_mode
        self.time_mode = tm
        ctype = control_params.get("type")
        if ctype == "pole_assignment":
            from .pole_assignment import PoleAssignmentController

            poles = control_params.get("poles") or []
            self.strategy = PoleAssignmentController(self.params, self.dt, tm, self.matrices_fn, poles)
        elif ctype == "state_feedback":
            from .state_feedback import StateFeedbackController

            gain = control_params.get("gain") or []
            self.strategy = StateFeedbackController(gain)
        elif ctype == "lqr":
            from .lqr_controller import LQRController

            q = control_params.get("Q") or []
            r = control_params.get("R") or []
            self.strategy = LQRController(self.params, self.dt, tm, self.matrices_fn, q, r)
        else:
            self.strategy = None


class _BaseControllerStrategy:
    def compute(self, state):
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def set_params(self, params) -> None:
        pass


class _LinearControllerStrategy(_BaseControllerStrategy):
    def __init__(self, params, dt: float, time_mode: str, matrices_fn: Callable[[Any], Tuple[np.ndarray, np.ndarray]]):
        self.params = params
        self.dt = dt
        self.time_mode = time_mode
        self.matrices_fn = matrices_fn
        self._rebuild_model()

    def _rebuild_model(self) -> None:
        self.A, self.B, self.Ad, self.Bd = build_linear_model(
            lambda: self.matrices_fn(self.params),
            self.dt,
            self.time_mode,
            include_output=False,
        )
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]

    def set_params(self, params) -> None:
        self.params = params
        self._rebuild_model()
