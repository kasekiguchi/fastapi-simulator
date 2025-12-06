# app/simulators/furutaPendulum/simulator.py
from dataclasses import dataclass
from typing import Any, Optional

from ..base import BaseSimulator, SimState

# ここは既存コードに合わせて import
# from .FURUTA_PENDULUM.base import (
    # FurutaPlant,      # 制御対象クラス (仮名)
    # PlantState,       # x を表す dataclass or class
    # PlantParams,      # 物理パラメータ
# )
# from .FURUTA_PENDULUM import apply_input  # x,u,dt -> x_next (仮)
# from .FURUTA_PENDULUM import measure      # x -> y
# from .FURUTA_PENDULUM import FURUTA_PENDULUM
from .FURUTA_PENDULUM import FURUTA_PENDULUM as fp

from .CONTROLLER import (
    CONTROLLER,
    # FurutaController,  # 制御器本体 (仮名)
    # ControllerState,   # z (内部状態)
    ControllerParams,  # 制御パラメータ
)
from .CONTROLLER import calc_input              # u = f(...)
from .ESTIMATOR import ESTIMATOR        # x̂ 更新 (必要なら)

from .common.state import FurutaPendulumState
from .common.physical_parameters import FurutaPendulumParams

class PublicFurutaState(SimState):
    """フロントへ送るための状態（必要な分だけ切り出し）"""
    theta: float = 0.0
    dtheta: float = 0.0
    phi: float = 0.0
    dphi: float = 0.0
    u: float = 0.0
    y0: float = 0.0
    y1: float = 0.0
    # デバッグ用にぜんぶ投げたいなら Any で持ってもOK（JSON化するとき注意）
    # raw_x: Any = None
    # raw_z: Any = None

class FurutaPendulumSimulator(BaseSimulator):
    """
    FURUTA_PENDULUM + CONTROLLER を BaseSimulator に接続するアダプタ。
    - plant: FURUTA_PENDULUM の制御対象
    - controller: CONTROLLER 側の制御器
    """

    def __init__(self, dt: float = 0.01,
        plant_params: Optional[FurutaPendulumParams] = None,
        controller_params: Optional[ControllerParams] = None,
        initial_state: Optional[FurutaPendulumState] = None,
        ):
        # dt
        self.dt = dt

        self.plant = fp(plant_params)
        # --- パラメータ: 渡されなければデフォルトを使う ---
        
        if plant_params is None:
            plant_params = self.plant.get_params()
        if controller_params is None:
            controller_params = ControllerParams()

        # --- 初期状態: 渡されなければゼロ状態などのデフォルト ---
        if initial_state is None:
            initial_state = FurutaPendulumState()
            
                # 内部に保持
        self.params: FurutaPendulumParams = plant_params
        self.state: FurutaPendulumState = initial_state
        self._t: float = initial_state.t if hasattr(initial_state, "t") else 0.0

        # プラント・コントローラ・推定器のインスタンス化（既存コードに合わせて調整）
        self.ctrl_params: ControllerParams = controller_params
        self.ctrl = FurutaController(self.ctrl_params)
        self.ctrl_state: ControllerState = self.ctrl.init_state()

        self._pending_impulse: float = 0.0  # クリック等で加える追加入力
        self._last_u: float = 0.0
        self._last_y: Any = None
        self.exp_mode: bool = False
        self.poles = []
        self.gain = []
#=================================================
        # プラント / コントローラの生成
        params = get_params()            # PlantParams を返す想定
        self.plant = FurutaPlant(params)         # あるいは PlantState, Params を渡すAPIでもOK
        self.x: PlantState = self.plant.state    # もしくは self.x = PlantState()
        self.y: Any = None

        self.estimator = ESTIMATOR(initial_state)
        self.ctrl_params = ControllerParams()    # ここも既存コードに合わせて
        self.ctrl = FurutaController(self.ctrl_params)
        self.z: ControllerState = self.ctrl.state  # 内部状態（オブザーバなど）

        # 前回入力
        self.u: float = 0.0

        # クリック等で加える追加入力
        self._pending_impulse: float = 0.0

        # 時刻は SimState 側で管理
        self._t: float = 0.0

    # ===== BaseSimulator インターフェース実装 =====
    def reset(self) -> None:
        """状態リセット（パラメータは維持）"""
        self.state = FurutaPendulumState()
        self.ctrl_state = self.ctrl.init_state()
        self._t = 0.0
        self._pending_impulse = 0.0
        self._last_u = 0.0
        self._last_y = None

    def set_params(self, **kwargs) -> None:
        """
        フロントからパラメータを部分的に更新するためのメソッド。
        例: { "plant_m1": 1.2, "plant_l1": 0.4, "ctrl_kp": 10.0 }
        """
        # plant_ で始まるものは物理パラメータへ
        for k, v in kwargs.items():
            if k.startswith("plant_"):
                name = k[len("plant_") :]
                if hasattr(self.params, name):
                    setattr(self.params, name, float(v))

        # ctrl_ で始まるものは制御パラメータへ
        for k, v in kwargs.items():
            if k.startswith("ctrl_"):
                name = k[len("ctrl_") :]
                if hasattr(self.ctrl_params, name):
                    setattr(self.ctrl_params, name, float(v))
    def set_poles(self, poles=None, gain=None) -> None:
        self.poles = poles or []
        self.gain = gain or []

    def set_exp_mode(self, exp_mode: bool) -> None:
        self.exp_mode = bool(exp_mode)

    def apply_impulse(self, **kwargs) -> None:
        """クリックなどで瞬間的に入れるトルク（追加入力）"""
        torque = float(kwargs.get("torque", 0.1))
        self._pending_impulse += torque

    def step(self) -> PublicFurutaState:
        dt = self.dt

        # 1) センサ値を取得
        y = measure(self.state)
        self._last_y = y

        # 2) 状態推定（必要な場合だけ）
        try:
            x_hat, self.ctrl_state = estimate(
                self.state, y, self._last_u, self.ctrl_state, dt
            )
        except NotImplementedError:
            x_hat = self.state

        # 3) 制御入力 u を計算
        u = calc_input(x_hat, self.ctrl_params, self.ctrl_state, t=self._t)

        # 4) クリックによる追加入力を加える
        if self._pending_impulse != 0.0:
            u += self._pending_impulse
            self._pending_impulse = 0.0

        self._last_u = u

        # 5) プラントを 1 ステップ進める
        self.state = apply_input_step(self.state, u, self.params, dt)

        # 6) 時刻更新
        self._t += dt

        # 7) 可視化用の PublicFurutaState に詰め直す
        theta = getattr(self.state, "theta", 0.0)
        dtheta = getattr(self.state, "dtheta", 0.0)
        phi = getattr(self.state, "phi", 0.0)
        dphi = getattr(self.state, "dphi", 0.0)

        if isinstance(y, (list, tuple)) and len(y) >= 2:
            y0, y1 = float(y[0]), float(y[1])
        else:
            y0 = float(y) if y is not None else 0.0
            y1 = 0.0

        return PublicFurutaState(
            t=self._t,
            theta=theta,
            dtheta=dtheta,
            phi=phi,
            dphi=dphi,
            u=u,
            y0=y0,
            y1=y1,
        )