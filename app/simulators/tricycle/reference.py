from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class RefSample:
    """リファレンス軌道のサンプル点"""
    pos: np.ndarray      # [x, y]
    theta: float          # 目標姿勢角
    v: float              # 目標並進速度
    alpha: float          # 目標操舵角


def _wrap_angle(a: float) -> float:
    """角度を [-pi, pi) に正規化"""
    return (a + math.pi) % (2 * math.pi) - math.pi


class TricycleReference:
    """三輪車モデルのリファレンス軌道基底クラス

    x軸時間軸制御に対応: sample(x) は車両の現在x座標に対応するリファレンス点を返す
    """

    def sample(self, x: float) -> RefSample:
        """車両のx座標に対応するリファレンス点を返す"""
        raise NotImplementedError

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TricycleReference":
        if not isinstance(data, dict):
            return HoldReference()
        rtype = data.get("type")
        # type未指定 or "point": x, y, theta から固定点リファレンスを生成
        if rtype is None or rtype == "point":
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            theta = float(data.get("theta", 0.0))
            return HoldReference(pos=(x, y), theta=theta)
        if rtype == "line":
            return LineReference(
                start=np.asarray(data.get("start", [0.0, 0.0]), dtype=float),
                end=np.asarray(data.get("end", [0.0, 0.0]), dtype=float),
                speed=float(data.get("speed", 0.0)),
            )
        return HoldReference()


class HoldReference(TricycleReference):
    """静止目標（固定点保持）"""

    def __init__(self, pos: Tuple[float, float] = (0.0, 0.0), theta: float = 0.0):
        self.pos = np.asarray(pos, dtype=float)
        self.theta = float(theta)

    def sample(self, x: float) -> RefSample:
        return RefSample(pos=self.pos, theta=self.theta, v=0.0, alpha=0.0)


class LineReference(TricycleReference):
    """直線軌道リファレンス（x座標ベース）

    始点から終点への直線上で、車両のx座標に対応するy, thetaを返す。
    """

    def __init__(self, start: np.ndarray, end: np.ndarray, speed: float):
        self.start = start.astype(float)
        self.end = end.astype(float)
        self.speed = float(speed)
        diff = self.end - self.start
        self.length = float(np.linalg.norm(diff))
        self.dir_vec = diff / self.length if self.length > 1e-9 else np.zeros_like(diff)
        self.theta = float(math.atan2(self.dir_vec[1], self.dir_vec[0])) if self.length > 0 else 0.0
        # x方向の範囲
        self.x_start = float(self.start[0])
        self.x_end = float(self.end[0])

    def sample(self, x: float) -> RefSample:
        """車両のx座標に対応する直線上の点を返す"""
        dx = self.x_end - self.x_start
        if abs(dx) < 1e-9:
            # x方向に動かない直線 → 始点を返す
            return RefSample(pos=self.start.copy(), theta=self.theta, v=self.speed, alpha=0.0)
        # x座標に基づく補間パラメータ (0〜1にクランプ)
        s = (x - self.x_start) / dx
        s = max(0.0, min(1.0, s))
        pos = self.start + self.dir_vec * (s * self.length)
        v = self.speed if 0.0 <= s < 1.0 else 0.0
        return RefSample(pos=pos, theta=self.theta, v=v, alpha=0.0)
