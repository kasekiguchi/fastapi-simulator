from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np

from ..base import BaseSimulator, SimState
from .common.parameters import SmdParams
from .common.state import SmdState
from .common.matrices import smd_matrices
from .CONTROLLER import CONTROLLER, ControllerParams
from .ESTIMATOR import ESTIMATOR, EstimatorParams


@dataclass
class PublicSmdState(SimState):
    t: float = 0.0
    p: float = 0.0
    v: float = 0.0
    u: float = 0.0
    y: float = 0.0
    closed_loop_poles: Optional[List[Dict[str, float]]] = None
    feedback_gain: Optional[List[List[float]]] = None


class SpringMassDamperSimulator(BaseSimulator):
    def __init__(
        self,
        dt: float = 0.01,
        params: Optional[SmdParams] = None,
        initial_state: Optional[SmdState] = None,
    ):
        self.dt = dt
        self.params = params or SmdParams()
        self.state = initial_state or SmdState()
        self._pending_impulse: float = 0.0
        self.controller: Optional[CONTROLLER] = None
        self.control_params: Dict[str, Any] = {}
        self.control_time_mode: str = "discrete"
        self.control_info: Dict[str, Any] = {}
        self.estimator: Optional[ESTIMATOR] = None
        self.estimator_params: Dict[str, Any] = {}
        self.estimator_time_mode: str = "discrete"
        self._est_state = None
        self._last_u: float = 0.0
        self._trace = self._empty_trace()

    def reset(self) -> None:
        self.state = SmdState()
        self._pending_impulse = 0.0
        self._trace = self._empty_trace()
        self._est_state = None
        self._last_u = 0.0
        if self.controller:
            self.controller.reset()
        if self.estimator:
            self.estimator.reset()

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))
        if self.controller:
            self.controller.set_params(**kwargs)
            self.control_info = self._compute_control_info()
        if self.estimator:
            self.estimator.set_params(**kwargs)

    def set_initial(self, **kwargs) -> None:
        if not kwargs:
            return
        self.state = SmdState(
            p=float(kwargs.get("p", kwargs.get("theta", 0.0))),
            v=float(kwargs.get("v", kwargs.get("dtheta", 0.0))),
        )
        self._trace = self._empty_trace()
        self._est_state = None

    def apply_impulse(self, **kwargs) -> None:
        force = float(kwargs.get("force", kwargs.get("torque", 0.0)))
        self._pending_impulse += force

    def set_control_params(self, control_params: Optional[Dict[str, Any]] = None) -> None:
        if not control_params or "type" not in control_params:
            self.controller = None
            self.control_params = {}
            self.control_info = {}
            return
        tm = control_params.get("timeMode") or control_params.get("time_mode") or self.control_time_mode
        control_params = {**control_params, "timeMode": tm, "time_mode": tm}
        self.control_params = control_params
        self.control_time_mode = tm
        settings = ControllerParams(time_mode=tm, dt=self.dt)
        self.controller = CONTROLLER(parameters=self.params, settings=settings, dt=self.dt)
        self.controller.set_control_params(control_params)
        self.control_info = self._compute_control_info()

    def set_estimator_params(self, estimator_params: Optional[Dict[str, Any]] = None) -> None:
        if not estimator_params or "type" not in estimator_params:
            self.estimator = None
            self.estimator_params = {}
            return
        tm = estimator_params.get("timeMode") or estimator_params.get("time_mode") or self.estimator_time_mode
        estimator_params = {**estimator_params, "timeMode": tm, "time_mode": tm}
        self.estimator_params = estimator_params
        self.estimator_time_mode = tm
        settings = EstimatorParams(time_mode=tm, dt=self.dt)
        self.estimator = ESTIMATOR(parameters=self.params, settings=settings, dt=self.dt)
        self.estimator.set_estimator_params(estimator_params)

    def step(self) -> PublicSmdState:
        u_ctrl = 0.0
        if self.controller:
            ctrl_state = self._est_state if (self._est_state is not None and not getattr(self.estimator, "passthrough", False)) else self.state.as_array
            try:
                u_ctrl = float(self.controller.compute(ctrl_state))
            except Exception:
                u_ctrl = 0.0
        u = u_ctrl + self._pending_impulse
        self._pending_impulse = 0.0
        if not np.isfinite(u):
            u = 0.0
        u = float(np.clip(u, -1e3, 1e3))  # 入力暴走ガード

        m, k, c = self.params.mass, self.params.k, self.params.c
        p, v = self.state.p, self.state.v
        dp = v
        dv = (u - c * v - k * p) / m
        self.state = SmdState(p=p + dp * self.dt, v=v + dv * self.dt, t=self.state.t + self.dt)
        y = self.state.p

        if self.estimator:
            try:
                est_state = self.estimator.estimate(self._last_u, np.array([y]))
            except Exception:
                est_state = None
            if est_state is None:
                est_state = self._est_state if self._est_state is not None else self.state
        else:
            est_state = self.state
        self._est_state = est_state
        self._last_u = u

        # trace
        self._trace["t"].append(self.state.t)
        self._trace["p"].append(self.state.p)
        self._trace["v"].append(self.state.v)
        self._trace["u"].append(u)
        self._trace["y"].append(y)
        if est_state is not None:
            if hasattr(est_state, "as_array"):
                self._trace["xh"].append(est_state.as_array.tolist())
            else:
                self._trace["xh"].append(np.asarray(est_state, dtype=float).tolist())
        else:
            self._trace["xh"].append(None)

        return PublicSmdState(
            t=self.state.t,
            p=self.state.p,
            v=self.state.v,
            u=u,
            y=y,
            closed_loop_poles=self.control_info.get("closed_loop_poles"),
            feedback_gain=self.control_info.get("feedback_gain"),
        )

    def get_public_state(self) -> PublicSmdState:
        return PublicSmdState(
            t=self.state.t,
            p=self.state.p,
            v=self.state.v,
            u=0.0,
            y=self.state.p,
            closed_loop_poles=self.control_info.get("closed_loop_poles"),
            feedback_gain=self.control_info.get("feedback_gain"),
        )

    def get_trace(self) -> Dict[str, Any]:
        trace = {k: v.copy() for k, v in self._trace.items()}
        trace["plot_order"] = ["p", "v", "xh"]
        return trace

    def _compute_control_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if not self.controller or not getattr(self.controller, "strategy", None):
            return info
        strat = self.controller.strategy
        Ad = getattr(strat, "Ad", None)
        Bd = getattr(strat, "Bd", None)
        A = getattr(strat, "A", None)
        B = getattr(strat, "B", None)
        tm = getattr(self.controller, "time_mode", "discrete")

        def eigs(Ac, Bc, K):
            try:
                return np.linalg.eigvals(Ac - Bc @ K)
            except Exception:
                return None

        # pole assignment / lqr / state_feedback
        K = None
        if hasattr(strat, "K"):
            try:
                K = np.asarray(strat.K, dtype=float)
                info["feedback_gain"] = K.tolist()
            except Exception:
                K = None
        elif hasattr(strat, "gain"):
            try:
                g = np.asarray(strat.gain, dtype=float).flatten()
                K = g.reshape(1, -1)
                info["feedback_gain"] = K.tolist()
            except Exception:
                K = None

        if K is not None:
            eigvals = None
            if tm == "discrete" and Ad is not None and Bd is not None:
                eigvals = eigs(Ad, Bd, K)
            if eigvals is None and A is not None and B is not None:
                eigvals = eigs(A, B, K)
            if eigvals is not None:
                info["closed_loop_poles"] = [{"re": float(ev.real), "im": float(ev.imag)} for ev in eigvals]
        return info

    @staticmethod
    def _empty_trace():
        return {"t": [], "p": [], "v": [], "u": [], "y": [], "xh": []}
