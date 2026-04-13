from dataclasses import dataclass

from ..base import SimState


@dataclass
class TractorTrailerState(SimState):
    """トラクタ・トレーラモデルの独立状態

    独立状態（シミュレーション対象）:
      - x2, y2    : トレーラ後車軸位置
      - theta1    : トラクタとトレーラの相対角 (tractor_heading - trailer_heading)
      - theta2    : トレーラ絶対姿勢角

    派生量（表示用）:
      - x1 = x2 + L2*cos(theta2), y1 = y2 + L2*sin(theta2)
      - tractor heading = theta1 + theta2
    """
    x2: float = 0.0
    y2: float = 0.0
    theta1: float = 0.0   # 相対角 (tractor - trailer)
    theta2: float = 0.0   # トレーラ絶対姿勢角
    v: float = 0.0        # トラクタ並進速度（制御入力として保持）
    alpha: float = 0.0    # トラクタ操舵角（制御入力として保持）
    t: float = 0.0


@dataclass
class TractorTrailerParams:
    """トラクタ・トレーラモデルの物理パラメータ"""
    L1: float = 0.4   # トラクタのホイールベース（後車軸～前輪）
    L2: float = 0.6   # トレーラ長（トレーラ後車軸～連結点、連結点はトラクタ後車軸）
