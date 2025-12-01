# Path: app/furuta_pendulum/simulator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, List

import numpy as np
from scipy import signal, linalg

from .FURUTA_PENDULUM import FURUTA_PENDULUM, FurutaPendulumParams
from .Ac import Ac_furutaPendulum
from .Bc import Bc_furutaPendulum

TimeMode = Literal["discrete", "continuous"]
EstimatorMode = Literal["EKF", "observer"]


@dataclass
# class SimConfig:
#     init: List[float]
#     dt: float = 0.01
#     duration: float = 10.0
#     time_mode: TimeMode = "discrete"
#     estimator: EstimatorMode = "observer"


class SimResult(TypedDict):
    t: List[float]
    y: List[List[float]]      # [p, th]
    xhat: List[List[float]]   # 推定状態
    u: List[float]


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = linalg.solve_discrete_are(A, B, Q, R)
    BtP = B.T @ P
    K = np.linalg.inv(BtP @ B + R) @ (BtP @ A)
    return K


def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = linalg.solve_continuous_are(A, B, Q, R)
    BtP = B.T @ P
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K


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
    # init = init
    # dt = dt
    te = duration
    tspan = np.arange(0.0, te + dt, dt)

    # プラント生成（MATLAB: cart = FURUTA_PENDULUM(init);）
    cart = FURUTA_PENDULUM(init, params=FurutaPendulumParams())

    # 線形モデル
    Ac = Ac_furutaPendulum(params=FurutaPendulumParams().as_array)
    Bc = Bc_furutaPendulum(params=FurutaPendulumParams().as_array)
    # Bc = np.array([[0], [0], [0], [1]], dtype=float)
    Cc = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    Dc = np.zeros((2, 1))

    if time_mode == "discrete":
        Ad, Bd, Cd, Dd, _ = signal.cont2discrete((Ac, Bc, Cc, Dc), dt)
    else:
        Ad, Bd, Cd, Dd = None, None, None, None

    Q_lqr = np.diag([1.0, 100.0, 1.0, 1.0])
    R_lqr = np.array([[1.0]])

    if time_mode == "discrete":
        Fd = dlqr(Ad, Bd, Q_lqr, R_lqr)
        Fod = dlqr(
            Ad.T,
            Cd.T,
            np.diag([1.0, 1.0, 1.0, 1.0]),
            np.diag([0.01,0.01]),
        ).T
        Fc = None
        Foc = None
    else:
        Fc = lqr(Ac, Bc, Q_lqr, R_lqr)
        Foc = lqr(
            Ac.T,
            Cc.T,
            np.diag([1.0, 1.0, 1.0, 1.0]),
            np.diag([0.01,0.01]),
        ).T
        Fd = None
        Fod = None

    if estimator == "EKF":
        P = np.eye(4)
        Qd = 1.0
        Rd = 0.01 * np.diag([0.02, 0.05])

    n_steps = len(tspan)
    T = np.zeros(n_steps)
    Y = np.zeros((n_steps, 2))
    Xhat = np.zeros((n_steps, 4))
    U = np.zeros(n_steps)

    # 初期測定 & 推定
    y = cart.measure()
    xh = np.array([y[0], y[1], 0.0, 0.0])
    u = 0.0

    for i, t in enumerate(tspan):
        T[i] = t

        # 測定
        y = cart.measure()

        # 推定
        if i > 0:
            if time_mode == "discrete":
                if estimator == "EKF":
                    xh_pre = Ad @ xh + Bd.flatten() * u
                    P_pre = Ad @ P @ Ad.T + Bd @ (Qd * Bd.T)
                    Gd = (P_pre @ Cd.T) @ np.linalg.inv(Cd @ P_pre @ Cd.T + Rd)
                    P = (np.eye(4) - Gd @ Cd) @ P_pre
                    xh = xh_pre + Gd @ (y - Cd @ xh_pre)
                else:
                    xh = Ad @ xh + Bd.flatten() * u - Fod @ (Cd @ xh - y)
            else:
                dxh = Ac @ xh + Bc.flatten() * u - Foc @ (Cc @ xh - y)
                xh = xh + dt * dxh

        # フィードバック
        if time_mode == "discrete":
            u = float(Fd @ (-xh))
        else:
            u = float(Fc @ (-xh))

        # プラントを進める
        cart.apply_input(u, dt)

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
