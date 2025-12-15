from dataclasses import dataclass
import numpy as np

from ...base import SimState


@dataclass
class SegwayState(SimState):
    """Segway状態: [x, vx, y, vy, phi, theta, psi, dphi, dtheta, dpsi]"""
    x: float = 0.0
    vx: float = 0.0
    y: float = 0.0
    vy: float = 0.0
    phi: float = 0.0          # ロール角
    theta: float = 0.0        # ピッチ角（倒立を0とする）
    psi: float = 0.0          # ヨー角
    dphi: float = 0.0         # ロール角速度
    dtheta: float = 0.0       # ピッチ角速度
    dpsi: float = 0.0         # ヨー角速度

    @property
    def as_array(self) -> np.ndarray:
        return np.array([
            self.x, self.vx, self.y, self.vy,
            self.phi, self.theta, self.psi,
            self.dphi, self.dtheta, self.dpsi
        ])

    def set(self, state):
        """配列から状態を設定"""
        self.x, self.vx, self.y, self.vy, self.phi, self.theta, self.psi, self.dphi, self.dtheta, self.dpsi = state
        return self


@dataclass
class SegwayParams:
    """Segway物理パラメータ"""
    M: float = 10.0      # ボディ質量 [kg]
    m: float = 1.0       # 搭乗者質量 [kg]
    Ixx: float = 0.2     # ロール方向慣性モーメント [kg·m^2]
    Iyy: float = 0.3     # ピッチ方向慣性モーメント [kg·m^2]
    Izz: float = 0.25    # ヨー方向慣性モーメント [kg·m^2]
    Iw: float = 0.01     # 車輪慣性モーメント [kg·m^2]
    r: float = 0.1       # 車輪半径 [m]
    L: float = 0.25      # 左右車輪間距離 [m]
    h: float = 0.3       # 質量中心の高さ [m]
    Damp: float = 0.05   # ダンピング係数
    gravity: float = 9.81  # 重力加速度 [m/s^2]
