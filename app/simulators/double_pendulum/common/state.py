from dataclasses import dataclass
import numpy as np

from ...base import SimState


@dataclass
class DoublePendulumState(SimState):
    """Double Pendulum状態: [x, vx, theta1, theta2, dtheta1, dtheta2]"""
    x: float = 0.0         # カート位置 [m]
    vx: float = 0.0        # カート速度 [m/s]
    theta1: float = 0.0    # 第1振り子角度 [rad]（上向き = 0）
    theta2: float = 0.0    # 第2振り子角度 [rad]（上向き = 0）
    dtheta1: float = 0.0   # 第1振り子角速度 [rad/s]
    dtheta2: float = 0.0   # 第2振り子角速度 [rad/s]

    @property
    def as_array(self) -> np.ndarray:
        return np.array([
            self.x, self.vx, self.theta1, self.theta2,
            self.dtheta1, self.dtheta2
        ])

    def set(self, state):
        """配列から状態を設定"""
        self.x, self.vx, self.theta1, self.theta2, self.dtheta1, self.dtheta2 = state
        return self


@dataclass
class DoublePendulumParams:
    """Double Pendulum物理パラメータ"""
    M: float = 5.0        # カート質量 [kg]
    m1: float = 0.5       # 第1振り子質量 [kg]
    m2: float = 0.3       # 第2振り子質量 [kg]
    L1: float = 0.5       # 第1振り子長さ [m]
    L2: float = 0.4       # 第2振り子長さ [m]
    lg1: float = 0.25     # 第1振り子重心距離 [m]
    lg2: float = 0.2      # 第2振り子重心距離 [m]
    I1: float = 0.01      # 第1振り子慣性モーメント [kg·m^2]
    I2: float = 0.008     # 第2振り子慣性モーメント [kg·m^2]
    friction: float = 0.01  # 摩擦係数
    gravity: float = 9.81   # 重力加速度 [m/s^2]
