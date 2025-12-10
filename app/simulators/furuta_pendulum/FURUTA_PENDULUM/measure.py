# Path: app/furuta_pendulum/FURUTA_PENDULUM/measure.py
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .base import FURUTA_PENDULUM


def measure(self: "FURUTA_PENDULUM", t: Optional[float] = None) -> np.ndarray:
    """
    MATLAB:
        output = obj.measure(t)
    の簡易版。
    ここでは t は無視して「現在の state から測定」を行う。
    （MATLAB のように TT, XX から補間したければ、ここに実装を追加）
    """
    self.output = self._H @ self.state
    noise = self.measure_noise * np.random.randn(2) if np.any(self.measure_noise) else 0.0
    return self.output + noise
