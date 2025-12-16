from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
from scipy.integrate import odeint

from ..base import BaseSimulator, SimState
from .common.state import SegwayState, SegwayParams
from .common.matrices import _segway_dynamics
from ..common.CONTROLLER.base import GenericController, ControllerParams


@dataclass
class PublicSegwayState(SimState):
    """フロントへ送るための公開状態"""
    t: float = 0.0
    x: float = 0.0
    vx: float = 0.0
    y: float = 0.0
    vy: float = 0.0
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    dphi: float = 0.0
    dtheta: float = 0.0
    dpsi: float = 0.0
    u: float = 0.0
    control_mode: str = "controller"
    closed_loop_poles: Optional[List[Dict[str, float]]] = None
    feedback_gain: Optional[List[List[float]]] = None
    design_error: Optional[str] = None


class SegwaySimulator(BaseSimulator):
    """Segway シミュレータ"""

    def __init__(
        self,
        dt: float = 0.01,
        params: Optional[SegwayParams] = None,
        initial_state: Optional[SegwayState] = None,
    ):
        self.dt = dt
        self.control_time_mode: str = "discrete"
        self.estimator_time_mode: str = "discrete"
        self.params: SegwayParams = params or SegwayParams()

        if initial_state is None:
            self._initial_array: np.ndarray = np.zeros(10)
        else:
            self._initial_array = initial_state.as_array

        self.state = SegwayState()
        self.state.set(self._initial_array.copy())

        self._pending_impulse: np.ndarray = np.zeros(2)  # [tau_left, tau_right]
        self.controller: Optional[GenericController] = None
        self.control_params: Dict[str, Any] = {}
        self.control_info: Dict[str, Any] = {}
        self._last_u: np.ndarray = np.zeros(2)
        self._trace: Dict[str, list] = self._empty_trace()

    def reset(self) -> None:
        """状態リセット"""
        self._pending_impulse = np.zeros(2)
        self.state = SegwayState()
        self.state.set(self._initial_array.copy())
        self._trace = self._empty_trace()
        self._last_u = np.zeros(2)
        if self.controller:
            self.controller.reset()

    def set_initial(self, initial=None, **kwargs) -> None:
        """初期状態を設定"""
        if initial is None and kwargs:
            initial = kwargs
        if initial is None or not isinstance(initial, dict):
            return

        state_arr = np.array([
            float(initial.get("x", 0.0)),
            float(initial.get("vx", 0.0)),
            float(initial.get("y", 0.0)),
            float(initial.get("vy", 0.0)),
            float(initial.get("phi", 0.0)),
            float(initial.get("theta", 0.0)),
            float(initial.get("psi", 0.0)),
            float(initial.get("dphi", 0.0)),
            float(initial.get("dtheta", 0.0)),
            float(initial.get("dpsi", 0.0)),
        ])
        self._initial_array = state_arr
        self.reset()

    def apply_impulse(self, **kwargs) -> None:
        """外部からのインパルス（クリック）を加える"""
        tau_left = float(kwargs.get("tau_left", kwargs.get("force", 0.0)))
        tau_right = float(kwargs.get("tau_right", 0.0))
        self._pending_impulse += np.array([tau_left, tau_right])

    def set_params(self, **kwargs) -> None:
        """物理パラメータを変更"""
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))
        if self.controller:
            self.control_info = self._compute_control_info()

    def step(self) -> PublicSegwayState:
        """1ステップ進める"""
        # 制御入力を計算
        control_input = self._compute_control_input()

        # インパルスを加える
        control_input += self._pending_impulse
        self._pending_impulse = np.zeros(2)

        # ODE を解く
        t_span = [0, self.dt]
        state_array = self.state.as_array
        
        def dynamics(y, t):
            return _segway_dynamics(y, control_input, self.params)

        solution = odeint(dynamics, state_array, t_span, full_output=False)
        new_state_array = solution[-1]

        # 状態を更新
        self.state.set(new_state_array)
        self.state.t += self.dt
        self._last_u = control_input.copy()

        # トレース記録
        self._record_trace()

        return self._get_public_state()

    def _compute_control_input(self) -> np.ndarray:
        """LQR制御を計算"""
        if not self.controller:
            return np.zeros(2)

        state_array = self.state.as_array
        try:
            # コントローラは単一スカラー出力を想定しているため、
            # ここではマトリクス形式で直接計算
            if hasattr(self.controller, 'strategy') and self.controller.strategy:
                K = self.controller.strategy.K
                u_scalar = -K @ state_array
                if isinstance(u_scalar, np.ndarray):
                    return u_scalar[:2] if len(u_scalar) >= 2 else np.array([float(u_scalar[0]), 0.0])
                else:
                    return np.array([float(u_scalar), 0.0])
            return np.zeros(2)
        except Exception:
            return np.zeros(2)

    def set_control_params(self, control_params: Optional[Dict[str, Any]] = None) -> None:
        """制御パラメータを設定"""
        if control_params is None or "type" not in control_params:
            self.controller = None
            return

        from .common.matrices import segway_linearized_matrices

        def matrices_fn(params):
            A, B, Ad, Bd = segway_linearized_matrices(params, self.dt)
            return (Ad if self.control_time_mode == "discrete" else A,
                    Bd if self.control_time_mode == "discrete" else B)

        settings = ControllerParams(time_mode=self.control_time_mode, dt=self.dt)
        self.controller = GenericController(
            self.params,
            matrices_fn,
            settings=settings,
            dt=self.dt
        )
        self.controller.set_control_params(control_params)
        self.control_info = self._compute_control_info()

    def _compute_control_info(self) -> Dict[str, Any]:
        """制御情報を計算"""
        info = {}
        if self.controller and self.controller.strategy:
            try:
                K = self.controller.strategy.K
                info["feedback_gain"] = K.tolist() if isinstance(K, np.ndarray) else K
                
                # クローズドループ極
                A = self.controller.strategy.A
                B = self.controller.strategy.B
                if A is not None and B is not None:
                    cl_eigs = np.linalg.eigvals(A - B @ K)
                    info["closed_loop_poles"] = [
                        {"re": float(eig.real), "im": float(eig.imag)} for eig in cl_eigs
                    ]
            except Exception:
                pass
        return info

    def _record_trace(self) -> None:
        """トレース記録"""
        state_dict = self.state.__dict__
        for key, value in state_dict.items():
            if key == "t":
                continue
            if key not in self._trace:
                self._trace[key] = []
            self._trace[key].append(float(value))
        if "t" not in self._trace:
            self._trace["t"] = []
        self._trace["t"].append(float(self.state.t))

    def get_public_state(self) -> PublicSegwayState:
        """公開状態を取得"""
        return self._get_public_state()

    def _get_public_state(self) -> PublicSegwayState:
        """内部用：公開状態オブジェクト生成"""
        return PublicSegwayState(
            t=self.state.t,
            x=self.state.x,
            vx=self.state.vx,
            y=self.state.y,
            vy=self.state.vy,
            phi=self.state.phi,
            theta=self.state.theta,
            psi=self.state.psi,
            dphi=self.state.dphi,
            dtheta=self.state.dtheta,
            dpsi=self.state.dpsi,
            u=float(self._last_u[0]) if len(self._last_u) > 0 else 0.0,
            control_mode=getattr(self, 'control_mode', 'controller'),
            closed_loop_poles=self.control_info.get("closed_loop_poles"),
            feedback_gain=self.control_info.get("feedback_gain"),
        )

    async def get_trace(self) -> Dict[str, list]:
        """トレース取得"""
        return self._trace

    def _empty_trace(self) -> Dict[str, list]:
        """空のトレース辞書"""
        return {
            "t": [], "x": [], "vx": [], "y": [], "vy": [],
            "phi": [], "theta": [], "psi": [],
            "dphi": [], "dtheta": [], "dpsi": []
        }
