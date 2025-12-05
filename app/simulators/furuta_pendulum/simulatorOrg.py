# Path: app/furuta_pendulum/simulator.py
from __future__ import annotations

from dataclasses import dataclass

from typing import Literal, TypedDict, List

import numpy as np

from .FURUTA_PENDULUM import FURUTA_PENDULUM
from .ESTIMATOR.base import ESTIMATOR
from .CONTROLLER.base import CONTROLLER

TimeMode = Literal["discrete", "continuous"]
EstimatorMode = Literal["EKF", "observer"]

@dataclass
class SimConfig:
    init: List[float]
    dt: float = 0.01
    duration: float = 10.0
    time_mode: TimeMode = "discrete"
    estimator: EstimatorMode = "observer"

class SimResult(TypedDict):
    t: List[float]
    y: List[List[float]]      # [p, th]
    xhat: List[List[float]]   # 推定状態
    u: List[float]

def simulate_furutaPendulum(
    init: List[float],
    dt: float = 0.01,
    duration: float = 10.0,
    time_mode: TimeMode = "discrete",
    estimator: EstimatorMode = "observer",
) -> SimResult:
    """
    MATLAB スクリプトの流れを Python で再現したもの。
    """
    init = init
    dt = dt
    te = duration
    tspan = np.arange(0.0, te + dt, dt)

    # プラント生成
    pend = FURUTA_PENDULUM(init, params=FurutaPendulumParams())
    param = pend.get_params()

    # データ保存
    n_steps = len(tspan)
    T = np.zeros(n_steps)
    Y = np.zeros((n_steps, 2))
    Xhat = np.zeros((n_steps, 4))
    U = np.zeros(n_steps)

    # 初期測定 & 推定器設定
    y = pend.measure()
    xh = np.array([y[0], y[1], 0.0, 0.0])
    if estimator == "EKF":
        P = np.eye(4)
        Qd = 1.0
        Rd = 0.01 * np.diag([0.02, 0.05])
    controller = CONTROLLER(param)
    estimator = ESTIMATOR("EKF",xh,P,Qd,Rd)
    u = 0.0

    # Simulation loop 
    for i, t in enumerate(tspan):
        T[i] = t

        # 測定
        y = pend.measure()
        u = controller.calc_input(xh)
        estimator.estimate(u,y)

        # プラントを進める
        pend.apply_input(u, dt)

        # ログ
        Y[i, :] = y
        Xhat[i, :] = xh
        U[i] = u

    return SimResult(
        t=T.tolist(),
        y=Y.tolist(),
        xhat=Xhat.tolist(),
        u=U.tolist(),
    )
