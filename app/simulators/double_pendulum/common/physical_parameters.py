"""Double Pendulum物理パラメータの定義"""

from dataclasses import dataclass


@dataclass
class DoublePendulumPhysicalParams:
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
