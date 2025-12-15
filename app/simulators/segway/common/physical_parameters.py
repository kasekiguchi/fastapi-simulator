"""Segway物理パラメータの定義（他の定義と共通インターフェース）"""

from dataclasses import dataclass


@dataclass
class SegwayPhysicalParams:
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
