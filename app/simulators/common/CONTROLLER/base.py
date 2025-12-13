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
        tm = control_params.get("timeMode") or control_params.get("time_mode") or self.time_mode
        control_params = {**control_params, "timeMode": tm, "time_mode": tm}
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
        elif ctype == "pid":
            from .pid import PIDController

            kp = control_params.get("kp", control_params.get("Kp", 0.0))
            ki = control_params.get("ki", control_params.get("Ki", 0.0))
            kd = control_params.get("kd", control_params.get("Kd", 0.0))
            ref = (
                control_params.get("ref")
                or control_params.get("reference")
                or control_params.get("refValues")
                or control_params.get("ref_values")
            )
            state_idx = control_params.get("stateIndex", control_params.get("state_index", 0))
            vel_idx = control_params.get("velIndex", control_params.get("vel_index", 1))
            ref_idx = control_params.get("refIndex", control_params.get("ref_index"))

            # Normalize indices: allow list/tuple [pos, vel]
            def _pair(val, default_pos, default_vel):
                if isinstance(val, (list, tuple)) and len(val) >= 1:
                    pos_i = val[0]
                    vel_i = val[1] if len(val) >= 2 else default_vel
                else:
                    pos_i = val
                    vel_i = default_vel
                return pos_i, vel_i

            pos_idx, vel_idx_norm = _pair(state_idx, 0, vel_idx)
            ref_pos_idx, ref_vel_idx = _pair(ref_idx, pos_idx, vel_idx_norm)
            self.strategy = PIDController(
                kp=kp,
                ki=ki,
                kd=kd,
                dt=self.dt,
                ref=ref if ref is not None else 0.0,
                pos_index=int(pos_idx) if pos_idx is not None else 0,
                vel_index=None if vel_idx_norm is None else int(vel_idx_norm),
                ref_pos_index=None if ref_pos_idx is None else int(ref_pos_idx),
                ref_vel_index=None if ref_vel_idx is None else int(ref_vel_idx),
            )
            if ref is not None:
                self.strategy.set_reference(ref)
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
