from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
from scipy.integrate import odeint

from ..base import BaseSimulator, SimState
from .common.state import DoublePendulumState, DoublePendulumParams
from .common.matrices import _double_pendulum_dynamics
from ..common.CONTROLLER.base import GenericController, ControllerParams


@dataclass
class PublicDoublePendulumState(SimState):
    """フロントへ送るための公開状態"""
    t: float = 0.0
    x: float = 0.0
    vx: float = 0.0
    theta1: float = 0.0
    theta2: float = 0.0
    dtheta1: float = 0.0
    dtheta2: float = 0.0
    u: float = 0.0
    control_mode: str = "controller"
    closed_loop_poles: Optional[List[Dict[str, float]]] = None
    feedback_gain: Optional[List[List[float]]] = None
    design_error: Optional[str] = None


class DoublePendulumSimulator(BaseSimulator):
    """Double Pendulum シミュレータ"""

    def __init__(
        self,
        dt: float = 0.01,
        params: Optional[DoublePendulumParams] = None,
        initial_state: Optional[DoublePendulumState] = None,
    ):
        self.dt = dt
        self.control_time_mode: str = "discrete"
        self.estimator_time_mode: str = "discrete"
        self.params: DoublePendulumParams = params or DoublePendulumParams()

        if initial_state is None:
            self._initial_array: np.ndarray = np.zeros(6)
        else:
            self._initial_array = initial_state.as_array

        self.state = DoublePendulumState()
        self.state.set(self._initial_array.copy())

        self._pending_impulse: float = 0.0  # F
        self.controller: Optional[GenericController] = None
        self.control_params: Dict[str, Any] = {}
        self.control_info: Dict[str, Any] = {}
        self._last_u: float = 0.0
        self._trace: Dict[str, list] = self._empty_trace()

    def reset(self) -> None:
        """状態リセット"""
        self._pending_impulse = 0.0
        self.state = DoublePendulumState()
        self.state.set(self._initial_array.copy())
        self._trace = self._empty_trace()
        self._last_u = 0.0
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
            float(initial.get("theta1", 0.0)),
            float(initial.get("theta2", 0.0)),
            float(initial.get("dtheta1", 0.0)),
            float(initial.get("dtheta2", 0.0)),
        ])
        self._initial_array = state_arr
        self.reset()

    def apply_impulse(self, **kwargs) -> None:
        """外部からのインパルス（クリック）を加える"""
        force = float(kwargs.get("force", kwargs.get("F", 0.0)))
        self._pending_impulse += force

    def set_params(self, **kwargs) -> None:
        """物理パラメータを変更"""
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))
        if self.controller:
            self.control_info = self._compute_control_info()

    def step(self) -> PublicDoublePendulumState:
        """1ステップ進める"""
        # 制御入力を計算
        control_input = self._compute_control_input()

        # インパルスを加える
        control_input += self._pending_impulse
        self._pending_impulse = 0.0

        # ODE を解く
        t_span = [0, self.dt]
        state_array = self.state.as_array
        
        def dynamics(y, t):
            return _double_pendulum_dynamics(y, np.array([control_input]), self.params)

        solution = odeint(dynamics, state_array, t_span, full_output=False)
        new_state_array = solution[-1]

        # 状態を更新
        self.state.set(new_state_array)
        self.state.t += self.dt
        self._last_u = control_input

        # トレース記録
        self._record_trace()

        return self._get_public_state()

    def _compute_control_input(self) -> float:
        """LQR制御を計算"""
        if not self.controller:
            return 0.0

        state_array = self.state.as_array
        try:
            if hasattr(self.controller, 'strategy') and self.controller.strategy:
                K = self.controller.strategy.K
                u_vec = -K @ state_array
                if isinstance(u_vec, np.ndarray):
                    return float(u_vec[0])
                else:
                    return float(u_vec)
            return 0.0
        except Exception:
            return 0.0

    def set_control_params(self, control_params: Optional[Dict[str, Any]] = None) -> None:
        """制御パラメータを設定"""
        if control_params is None or "type" not in control_params:
            self.controller = None
            return

        from .common.matrices import double_pendulum_linearized_matrices

        def matrices_fn(params):
            A, B, Ad, Bd = double_pendulum_linearized_matrices(params, self.dt)
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

    def get_public_state(self) -> PublicDoublePendulumState:
        """公開状態を取得"""
        return self._get_public_state()

    def _get_public_state(self) -> PublicDoublePendulumState:
        """内部用：公開状態オブジェクト生成"""
        return PublicDoublePendulumState(
            t=self.state.t,
            x=self.state.x,
            vx=self.state.vx,
            theta1=self.state.theta1,
            theta2=self.state.theta2,
            dtheta1=self.state.dtheta1,
            dtheta2=self.state.dtheta2,
            u=self._last_u,
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
            "t": [], "x": [], "vx": [], "theta1": [], "theta2": [],
            "dtheta1": [], "dtheta2": []
        }
