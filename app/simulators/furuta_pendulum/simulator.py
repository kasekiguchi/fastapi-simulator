from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Sequence

import numpy as np

from ..base import BaseSimulator, SimState
from .FURUTA_PENDULUM.base import FURUTA_PENDULUM
from .common.state import FurutaPendulumState
from .common.physical_parameters import FurutaPendulumParams
from .CONTROLLER import CONTROLLER, ControllerParams
from .ESTIMATOR import ESTIMATOR, EstimatorParams
from ..common.linear_utils import build_linear_model


@dataclass
class PublicFPState(SimState):
    """フロントへ送るための公開状態"""
    t: float = 0.0
    theta: float = 0.0
    dtheta: float = 0.0
    phi: float = 0.0
    dphi: float = 0.0
    u: float = 0.0
    y0: float = 0.0
    y1: float = 0.0
    closed_loop_poles: Optional[List[Dict[str, float]]] = None
    feedback_gain: Optional[List[List[float]]] = None
    design_error: Optional[str] = None


class FurutaPendulumSimulator(BaseSimulator):
    """
    FURUTA_PENDULUM を BaseSimulator に接続するシンプルなラッパ。
    """

    def __init__(
        self,
        dt: float = 0.01,
        plant_params: Optional[FurutaPendulumParams] = None,
        initial_state: Optional[FurutaPendulumState] = None,
    ):
        self.dt = dt
        self.control_time_mode: str = "discrete"
        self.estimator_time_mode: str = "discrete"
        self.params: FurutaPendulumParams = plant_params or FurutaPendulumParams()

        if initial_state is None:
            self._initial_array: Sequence[float] = [0.0, 0.0, 0.0, 0.0]
        else:
            self._initial_array = initial_state.as_array

        self.plant = FURUTA_PENDULUM(initial=self._initial_array, params=self.params)
        self.state = FurutaPendulumState()
        self._pending_impulse: float = 0.0
        self.exp_mode: bool = False
        self.controller: Optional[CONTROLLER] = None
        self.control_params: Dict[str, Any] = {}
        self.control_info: Dict[str, Any] = {}
        self.estimator: Optional[ESTIMATOR] = None
        self.estimator_params: Dict[str, Any] = {}
        self._est_state: Optional[Any] = None
        self._last_u: float = 0.0
        self._trace: Dict[str, list] = self._empty_trace()

    def reset(self) -> None:
        """状態リセット（パラメータは維持）"""
        self._pending_impulse = 0.0
        self.plant = FURUTA_PENDULUM(initial=self._initial_array, params=self.params)
        self.state = FurutaPendulumState()
        self._trace = self._empty_trace()
        self._est_state = None
        self._last_u = 0.0
        if self.controller:
            self.controller.reset()
        if self.estimator:
            self.estimator.reset()

    def set_initial(self, initial=None, **kwargs) -> None:
        """初期状態を設定し直す。initial dict または kwargs で指定。"""
        if initial is None and kwargs:
            initial = kwargs
        if initial is None:
            return
        arr = [
            float(initial.get("theta", 0.0)),
            float(initial.get("phi", 0.0)),
            float(initial.get("dtheta", 0.0)),
            float(initial.get("dphi", 0.0)),
        ]
        self._initial_array = arr
        self.plant = FURUTA_PENDULUM(initial=self._initial_array, params=self.params)
        self.state = FurutaPendulumState(theta=arr[0], phi=arr[1], dtheta=arr[2], dphi=arr[3])
        self._trace = self._empty_trace()
        self._est_state = None
        self._last_u = 0.0

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))
        # FURUTA_PENDULUM が参照するパラメータも更新
        self.plant.params = self.params
        self.plant.plant_param = self.params.as_array
        if self.controller:
            self.controller.set_params(**kwargs)
            self.control_info = self._compute_control_info()
        if self.estimator:
            self.estimator.set_params(**kwargs)

    def set_control_params(self, control_params: Optional[Dict[str, Any]] = None) -> None:
        """制御器を選択し、パラメータを設定する。"""
        if not control_params or "type" not in control_params:
            self.controller = None
            self.control_params = {}
            self.control_info = {}
            return

        self.control_params = control_params
        tm = control_params.get("time_mode") or control_params.get("timeMode") or self.control_time_mode
        self.control_time_mode = tm
        settings = ControllerParams(time_mode=tm, dt=self.dt)
        self.controller = CONTROLLER(parameters=self.params, settings=settings, dt=self.dt)
        self.controller.set_control_params(control_params)
        self.control_info = self._compute_control_info()

    def set_estimator_params(self, estimator_params: Optional[Dict[str, Any]] = None) -> None:
        """Estimator を選択し、パラメータを設定する。"""
        if not estimator_params or "type" not in estimator_params:
            self.estimator = None
            self.estimator_params = {}
            return

        self.estimator_params = estimator_params
        tm = estimator_params.get("time_mode") or estimator_params.get("timeMode") or self.estimator_time_mode
        self.estimator_time_mode = tm
        settings = EstimatorParams(time_mode=tm, dt=self.dt)
        self.estimator = ESTIMATOR(parameters=self.params, settings=settings, dt=self.dt)
        self.estimator.set_estimator_params(estimator_params)

    def set_exp_mode(self, exp_mode: bool) -> None:
        self.exp_mode = bool(exp_mode)

    def apply_impulse(self, **kwargs) -> None:
        """クリックなどで瞬間的に入れるトルク（追加入力）"""
        torque = kwargs.get("torque", kwargs.get("force", 0.1))
        try:
            torque = float(torque)
        except Exception:
            torque = 0.0
        self._pending_impulse += torque

    def step(self) -> PublicFPState:
        # 最新計測で推定更新（制御に推定値を使う）
        measurement = self.plant.measure()
        if self.estimator:
            try:
                est_state = self.estimator.estimate(self._last_u, measurement)
            except Exception:
                est_state = None
            if est_state is None:
                est_state = self._est_state if self._est_state is not None else self.plant.state
        # else:
        #     est_state = self.plant.state
        self._est_state = est_state
        print(f"{est_state}===============")
        # 制御器があれば推定状態から入力を計算
        u_ctrl = 0.0
        if self.controller:
            ctrl_state = self._est_state if (self._est_state is not None and not getattr(self.estimator, "passthrough", False))            else self.plant.state
            try:
                u_ctrl = float(self.controller.compute(ctrl_state))
            except Exception:
                u_ctrl = 0.0

        # クリック等で加える追加入力
        u = u_ctrl + self._pending_impulse
        # クリック入力は1ステップ分のみ有効
        self._pending_impulse = 0.0
        if not np.isfinite(u):
            u = 0.0
        u = float(np.clip(u, -1e3, 1e3))  # 入力暴走の簡易ガード

        # プラントを進める
        try:
            plant_state = self.plant.apply_input(u, self.dt)
        except Exception as e:
            print("[FurutaPendulumSimulator] apply_input error:", e)
            plant_state = self.plant.state
        try:
            measurement = self.plant.measure()
        except Exception as e:
            print("[FurutaPendulumSimulator] measure error:", e)
            measurement = np.zeros(2)
        self._last_u = u
        theta = float(plant_state[0]) if len(plant_state) > 0 else 0.0
        phi = float(plant_state[1]) if len(plant_state) > 1 else 0.0
        dtheta = float(plant_state[2]) if len(plant_state) > 2 else 0.0
        dphi = float(plant_state[3]) if len(plant_state) > 3 else 0.0

        y0 = float(measurement[0]) if len(measurement) >= 1 else 0.0
        y1 = float(measurement[1]) if len(measurement) >= 2 else 0.0

        if self.estimator:
            try:
                est_state = self.estimator.estimate(u, measurement)
            except Exception:
                est_state = None
            # Passthrough or failed estimate -> fall back to previous estimate or true state
            if est_state is None:
                est_state = self._est_state if self._est_state is not None else self.plant.state
        else:
            est_state = self.plant.state
        self._est_state = est_state

        # Trace logging
        self._trace["t"].append(self.plant.t)
        self._trace["theta"].append(theta)
        self._trace["phi"].append(phi)
        self._trace["dtheta"].append(dtheta)
        self._trace["dphi"].append(dphi)
        self._trace["u"].append(u)
        self._trace["y0"].append(y0)
        self._trace["y1"].append(y1)
        if est_state is not None:
            if hasattr(est_state, "as_array"):
                xh = np.asarray(est_state.as_array, dtype=float).flatten()
            else:
                xh = np.asarray(est_state, dtype=float).flatten()
            self._trace["xh"].append(xh.tolist())
            print(f"==================={xh}")
        else:
            self._trace["xh"].append(None)

        theta = float(plant_state[0]) if len(plant_state) > 0 else 0.0
        phi = float(plant_state[1]) if len(plant_state) > 1 else 0.0
        dtheta = float(plant_state[2]) if len(plant_state) > 2 else 0.0
        dphi = float(plant_state[3]) if len(plant_state) > 3 else 0.0

        y0 = float(measurement[0]) if len(measurement) >= 1 else 0.0
        y1 = float(measurement[1]) if len(measurement) >= 2 else 0.0

        return PublicFPState(
            t=self.plant.t,
            theta=theta,
            dtheta=dtheta,
            phi=phi,
            dphi=dphi,
            u=u,
            y0=y0,
            y1=y1,
            closed_loop_poles=self.control_info.get("closed_loop_poles"),
            feedback_gain=self.control_info.get("feedback_gain"),
            design_error=self.control_info.get("design_error"),
        )

    def get_public_state(self) -> PublicFPState:
        """現在のプラント状態を返す（ステップを進めない）。"""
        plant_state = self.plant.state
        measurement = self.plant.measure()

        theta = float(plant_state[0]) if len(plant_state) > 0 else 0.0
        phi = float(plant_state[1]) if len(plant_state) > 1 else 0.0
        dtheta = float(plant_state[2]) if len(plant_state) > 2 else 0.0
        dphi = float(plant_state[3]) if len(plant_state) > 3 else 0.0

        y0 = float(measurement[0]) if len(measurement) >= 1 else 0.0
        y1 = float(measurement[1]) if len(measurement) >= 2 else 0.0

        return PublicFPState(
            t=self.plant.t,
            theta=theta,
            dtheta=dtheta,
            phi=phi,
            dphi=dphi,
            u=self._pending_impulse,
            y0=y0,
            y1=y1,
            closed_loop_poles=self.control_info.get("closed_loop_poles"),
            feedback_gain=self.control_info.get("feedback_gain"),
            design_error=self.control_info.get("design_error"),
        )

    def get_trace(self) -> Dict[str, Any]:
        """Return accumulated simulation trace."""
        trace = {k: v.copy() for k, v in self._trace.items()}
        trace["plot_order"] = ["theta", "phi", "dtheta", "dphi", "xh"]
        return trace

    def _compute_control_info(self) -> Dict[str, Any]:
        """Compute diagnostic info (closed-loop poles or gain) for UI."""
        info: Dict[str, Any] = {}
        if not self.controller or not getattr(self.controller, "strategy", None):
            return info

        ctype = self.control_params.get("type")
        strat = self.controller.strategy

        # Prefer the model inside the strategy if available
        Ad = getattr(strat, "Ad", None)
        Bd = getattr(strat, "Bd", None)
        if Ad is None or Bd is None:
            try:
                _, _, Ad, Bd = build_linear_model(
                    lambda: (self.controller.matrices_fn(self.params)),  # type: ignore[attr-defined]
                    self.dt,
                    getattr(self.controller, "time_mode", "discrete"),
                    include_output=False,
                )
            except Exception:
                Ad, Bd = None, None

        # Handle pole assignment: expose gain and closed-loop poles if possible
        if ctype == "pole_assignment" and hasattr(strat, "K"):
            try:
                K = np.asarray(strat.K, dtype=float)
                info["feedback_gain"] = K.tolist()
            except Exception:
                info["feedback_gain"] = None
                K = None
            if hasattr(strat, "design_error") and strat.design_error:
                info["design_error"] = strat.design_error
            if Ad is not None and Bd is not None and K is not None:
                try:
                    Acl = Ad - Bd @ K
                    eigvals = np.linalg.eigvals(Acl)
                    info["closed_loop_poles"] = [
                        {"re": float(ev.real), "im": float(ev.imag)} for ev in eigvals
                    ]
                except Exception:
                    info["closed_loop_poles"] = None
            return info

        # For LQR / state_feedback, expose closed-loop poles
        if ctype in ("lqr", "state_feedback") and (hasattr(strat, "gain") or hasattr(strat, "K")):
            # Get gain as 2D matrix
            K = None
            if hasattr(strat, "K"):
                try:
                    K = np.asarray(strat.K, dtype=float)
                except Exception:
                    K = None
            if K is None and hasattr(strat, "gain"):
                try:
                    g = np.asarray(strat.gain, dtype=float).flatten()
                    K = g.reshape(1, -1)
                except Exception:
                    K = None

            if Ad is None or Bd is None or K is None:
                return info

            try:
                Acl = Ad - Bd @ K
                eigvals = np.linalg.eigvals(Acl)
                info["closed_loop_poles"] = [
                    {"re": float(ev.real), "im": float(ev.imag)} for ev in eigvals
                ]
            except Exception:
                info["closed_loop_poles"] = None
        return info

    @staticmethod
    def _empty_trace() -> Dict[str, list]:
        return {
            "t": [],
            "theta": [],
            "phi": [],
            "dtheta": [],
            "dphi": [],
            "u": [],
            "y0": [],
            "y1": [],
            "xh": [],
        }
