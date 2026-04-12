from dataclasses import dataclass

from ..base import SimState


@dataclass
class TricycleState(SimState):
    """三輪車モデルの状態 [x, y, theta]"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0       # 並進速度（制御入力として保持）
    alpha: float = 0.0   # 操舵角（制御入力として保持）


@dataclass
class TricycleParams:
    """三輪車モデルのパラメータ"""
    L: float = 0.4  # ホイールベース（後輪軸中心から前輪まで）
