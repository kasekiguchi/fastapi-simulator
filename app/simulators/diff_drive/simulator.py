from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np

from ..base import BaseSimulator, SimState
from .state import DiffDriveParams, DiffDriveState
from .reference import DiffDriveReference, RefSample, HoldReference
from .controller import DiffDriveController


@dataclass
class PublicDiffDriveState(SimState):
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    ref_x: float = 0.0
    ref_y: float = 0.0
    ref_theta: float = 0.0


class DiffDriveSimulator(BaseSimulator):
    def __init__(self, dt: float = 0.02, params: Optional[DiffDriveParams] = None):
        self.dt = dt
        self.params = params or DiffDriveParams()
        self._initial_state = DiffDriveState()
        self.state = DiffDriveState(
            x=self._initial_state.x,
            y=self._initial_state.y,
            theta=self._initial_state.theta,
            v=self._initial_state.v,
            omega=self._initial_state.omega,
            t=0.0,
        )
        self.controller = DiffDriveController(self.params)
        self.reference: DiffDriveReference = HoldReference()
        self.control_mode: str = "controller"  # "controller" | "external"
        self._external_input: tuple[float, float] = (0.0, 0.0)
        self._pending_v: float = 0.0
        self._pending_omega: float = 0.0
        self._last_control = (0.0, 0.0)
        self._trace: Dict[str, list] = self._empty_trace()

    def reset(self) -> None:
        self.state = DiffDriveState(
            x=self._initial_state.x,
            y=self._initial_state.y,
            theta=self._initial_state.theta,
            v=self._initial_state.v,
            omega=self._initial_state.omega,
            t=0.0,
        )
        self._pending_v = 0.0
        self._pending_omega = 0.0
        self._last_control = (0.0, 0.0)
        self._trace = self._empty_trace()

    def set_initial(self, initial=None, **kwargs) -> None:
        if initial is None and kwargs:
            initial = kwargs
        if not isinstance(initial, dict):
            return
        x = float(initial.get("x", 0.0))
        y = float(initial.get("y", 0.0))
        theta = float(initial.get("theta", 0.0))
        v = float(initial.get("v", 0.0))
        omega = float(initial.get("omega", 0.0))
        self._initial_state = DiffDriveState(x=x, y=y, theta=theta, v=v, omega=omega, t=0.0)
        self.reset()

    def apply_impulse(self, **kwargs) -> None:
        self._pending_v += float(kwargs.get("v", kwargs.get("force", 0.0)))
        self._pending_omega += float(kwargs.get("omega", kwargs.get("torque", 0.0)))

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))

    def set_control_params(self, control_params: Optional[Dict[str, Any]] = None) -> None:
        if not control_params:
            return
        ctype = control_params.get("type") if isinstance(control_params, dict) else None
        if ctype == "external":
            inp = control_params.get("input") if isinstance(control_params, dict) else None
            v, omega = 0.0, 0.0
            if isinstance(inp, (list, tuple)) and len(inp) >= 2:
                v, omega = float(inp[0]), float(inp[1])
            elif isinstance(inp, dict):
                v = float(inp.get("v", 0.0))
                omega = float(inp.get("omega", 0.0))
            self._external_input = (v, omega)
            self.control_mode = "external"
            return

        self.control_mode = "controller"
        self.controller.set_control_params(control_params)

    def set_reference(self, reference: Optional[Dict[str, Any]]) -> None:
        if reference is None:
            self.reference = HoldReference()
        else:
            self.reference = DiffDriveReference.from_dict(reference)

    def _integrate(self, v: float, omega: float) -> None:
        x, y, theta = self.state.x, self.state.y, self.state.theta
        x = x + self.dt * v * np.cos(theta)
        y = y + self.dt * v * np.sin(theta)
        theta = theta + self.dt * omega
        self.state = DiffDriveState(x=x, y=y, theta=theta, v=v, omega=omega, t=self.state.t + self.dt)

    def _compute_control(self, ref: RefSample) -> tuple[float, float]:
        if self.controller:
            return self.controller.compute(self.state, ref)
        return ref.v, ref.omega

    def step(self) -> PublicDiffDriveState:
        ref = self.reference.sample(self.state.t) if self.reference else RefSample(np.zeros(2), 0.0, 0.0, 0.0)
        if self.control_mode == "external":
            v_cmd, omega_cmd = self._external_input
        else:
            v_cmd, omega_cmd = self._compute_control(ref)
        v_cmd += self._pending_v
        omega_cmd += self._pending_omega
        self._pending_v = 0.0
        self._pending_omega = 0.0
        self._last_control = (v_cmd, omega_cmd)
        self._integrate(v_cmd, omega_cmd)
        self._log_trace(ref)
        return PublicDiffDriveState(
            t=self.state.t,
            x=self.state.x,
            y=self.state.y,
            theta=self.state.theta,
            v=self.state.v,
            omega=self.state.omega,
            ref_x=ref.pos[0],
            ref_y=ref.pos[1],
            ref_theta=ref.theta,
        )

    def get_public_state(self) -> PublicDiffDriveState:
        ref = self.reference.sample(self.state.t) if self.reference else RefSample(np.zeros(2), 0.0, 0.0, 0.0)
        return PublicDiffDriveState(
            t=self.state.t,
            x=self.state.x,
            y=self.state.y,
            theta=self.state.theta,
            v=self.state.v,
            omega=self.state.omega,
            ref_x=ref.pos[0],
            ref_y=ref.pos[1],
            ref_theta=ref.theta,
        )

    def get_trace(self) -> Dict[str, list]:
        return {k: v.copy() for k, v in self._trace.items()}

    @staticmethod
    def _empty_trace() -> Dict[str, list]:
        return {"t": [], "x": [], "y": [], "theta": [], "v": [], "omega": [], "ref": []}

    def _log_trace(self, ref: RefSample) -> None:
        self._trace["t"].append(self.state.t)
        self._trace["x"].append(self.state.x)
        self._trace["y"].append(self.state.y)
        self._trace["theta"].append(self.state.theta)
        self._trace["v"].append(self.state.v)
        self._trace["omega"].append(self.state.omega)
        self._trace["ref"].append(
            {"x": float(ref.pos[0]), "y": float(ref.pos[1]), "theta": float(ref.theta)}
        )
