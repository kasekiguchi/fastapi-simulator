from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class RefSample:
    pos: np.ndarray
    yaw: float
    v: float
    omega: float


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class DiffDriveReference:
    def sample(self, t: float) -> RefSample:
        raise NotImplementedError

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DiffDriveReference":
        if not isinstance(data, dict):
            return HoldReference()
        rtype = data.get("type")
        if rtype == "line":
            return LineReference(
                start=np.asarray(data.get("start", [0.0, 0.0]), dtype=float),
                end=np.asarray(data.get("end", [0.0, 0.0]), dtype=float),
                speed=float(data.get("speed", 0.0)),
            )
        if rtype == "circle":
            return CircleReference(
                center=np.asarray(data.get("center", [0.0, 0.0]), dtype=float),
                radius=float(data.get("radius", 1.0)),
                speed=float(data.get("speed", 0.0)),
                clockwise=bool(data.get("clockwise", False)),
            )
        if rtype == "ellipse":
            return EllipseReference(
                center=np.asarray(data.get("center", [0.0, 0.0]), dtype=float),
                rx=float(data.get("rx", 1.0)),
                ry=float(data.get("ry", 1.0)),
                speed=float(data.get("speed", 0.0)),
                clockwise=bool(data.get("clockwise", False)),
            )
        return HoldReference()


class HoldReference(DiffDriveReference):
    def __init__(self, pos: Tuple[float, float] = (0.0, 0.0), yaw: float = 0.0):
        self.pos = np.asarray(pos, dtype=float)
        self.yaw = float(yaw)

    def sample(self, t: float) -> RefSample:
        return RefSample(pos=self.pos, yaw=self.yaw, v=0.0, omega=0.0)


class LineReference(DiffDriveReference):
    def __init__(self, start: np.ndarray, end: np.ndarray, speed: float):
        self.start = start.astype(float)
        self.end = end.astype(float)
        self.speed = float(speed)
        diff = self.end - self.start
        self.length = float(np.linalg.norm(diff))
        self.dir_vec = diff / self.length if self.length > 1e-9 else np.zeros_like(diff)
        self.yaw = float(math.atan2(self.dir_vec[1], self.dir_vec[0])) if self.length > 0 else 0.0

    def sample(self, t: float) -> RefSample:
        if self.length <= 0:
            return RefSample(pos=self.start, yaw=self.yaw, v=0.0, omega=0.0)
        s = max(0.0, min(self.speed * t, self.length))
        pos = self.start + self.dir_vec * s
        v = self.speed if s < self.length else 0.0
        return RefSample(pos=pos, yaw=self.yaw, v=v, omega=0.0)


class CircleReference(DiffDriveReference):
    def __init__(self, center: np.ndarray, radius: float, speed: float, clockwise: bool = False):
        self.center = center.astype(float)
        self.radius = max(radius, 1e-6)
        self.speed = float(speed)
        self.dir = -1.0 if clockwise else 1.0

    def sample(self, t: float) -> RefSample:
        ang_vel = self.dir * (self.speed / self.radius if self.radius > 0 else 0.0)
        ang = ang_vel * t
        pos = self.center + np.array([self.radius * math.cos(ang), self.radius * math.sin(ang)], dtype=float)
        xdot = -self.radius * math.sin(ang) * ang_vel
        ydot = self.radius * math.cos(ang) * ang_vel
        yaw = math.atan2(ydot, xdot)
        v = math.hypot(xdot, ydot)
        omega = ang_vel
        return RefSample(pos=pos, yaw=_wrap_angle(yaw), v=v, omega=omega)


class EllipseReference(DiffDriveReference):
    def __init__(self, center: np.ndarray, rx: float, ry: float, speed: float, clockwise: bool = False):
        self.center = center.astype(float)
        self.rx = max(rx, 1e-6)
        self.ry = max(ry, 1e-6)
        self.speed = float(speed)
        self.dir = -1.0 if clockwise else 1.0
        self._norm_radius = 0.5 * (self.rx + self.ry)

    def sample(self, t: float) -> RefSample:
        ang_vel = self.dir * (self.speed / self._norm_radius if self._norm_radius > 0 else 0.0)
        ang = ang_vel * t
        pos = self.center + np.array([self.rx * math.cos(ang), self.ry * math.sin(ang)], dtype=float)
        xdot = -self.rx * math.sin(ang) * ang_vel
        ydot = self.ry * math.cos(ang) * ang_vel
        v = math.hypot(xdot, ydot)
        yaw = math.atan2(ydot, xdot)
        omega = ang_vel
        return RefSample(pos=pos, yaw=_wrap_angle(yaw), v=v, omega=omega)
