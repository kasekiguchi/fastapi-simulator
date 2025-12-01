# Path: app/furuta_pendulum/FURUTA_PENDULUM/ode.py
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .base import FURUTA_PENDULUM


def ode(self: "FURUTA_PENDULUM", x: np.ndarray, u: float, param: np.ndarray) -> np.ndarray:
    """
    MATLAB の private メソッド ode(obj,in1,input,in3) に相当。

    ここに、Symbolic Math Toolbox が生成した t2=cos(p) ... の超長い式を
    Python に書き換えて貼り付けることができます。

    現在は接続確認用にかなり単純化したモデルを使っています。
    """
    p, th, dp, dth = x
    (
        m1,
        m2,
        J,
        jx,
        jy,
        jz,
        L,
        lg,
        Dp,
        Dth,
        gravity,
        a,
    ) = param

    # ---- TODO: ここを MATLAB から本物の式に差し替え ----
    # 簡易な安定化モデル（テスト用）
    k_p = 1.0
    k_th = 1.0

    ddp = -k_p * p + a * u - Dp * dp
    ddth = -k_th * th - Dth * dth

    return np.array([dp, dth, ddp, ddth], dtype=float)
