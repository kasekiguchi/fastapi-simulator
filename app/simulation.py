# Path: app/simulation.py
from dataclasses import dataclass
from typing import Literal, TypedDict, List

import numpy as np
from scipy import signal, linalg

# ==== 前提: ここはあなたが既に Python 版を持っている想定 ====
# from your_module import FurutaPendulum, ACCESS_PLANT, Ac_FurutaPendulum, Bc_FurutaPendulum

TimeMode = Literal["discrete", "continuous"]
EstimatorMode = Literal["EKF", "observer"]


@dataclass
class SimConfig:
    dt: float = 0.01
    duration: float = 10.0
    time_mode: TimeMode = "discrete"
    estimator: EstimatorMode = "observer"


class SimResult(TypedDict):
    t: List[float]          # 時刻列
    y: List[List[float]]    # 出力 [theta1, theta2]
    xhat: List[List[float]] # 推定状態 [theta1, theta2, dtheta1, dtheta2]
    u: List[float]          # 制御入力


def dlqr(A: np.ndarray, B: np.ndarray,
         Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """MATLAB の dlqr とほぼ同じもの (離散 LQR)"""
    P = linalg.solve_discrete_are(A, B, Q, R)
    BtP = B.T @ P
    K = np.linalg.inv(BtP @ B + R) @ (BtP @ A)
    return K


def lqr(A: np.ndarray, B: np.ndarray,
        Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """MATLAB の lqr とほぼ同じもの (連続 LQR)"""
    P = linalg.solve_continuous_are(A, B, Q, R)
    BtP = B.T @ P
    K = np.linalg.inv(BtP @ B + R) @ (BtP @ A)
    return K


def simulate_furuta(
    init: np.ndarray,
    config: SimConfig,
) -> SimResult:
    """
    MATLAB スクリプトのロジックを簡略化して Python に移植したもの。
    - init: 初期状態 [theta1, theta2, dtheta1, dtheta2]
    - config: サンプリング周期dt, 時間長さduration, estimatorモードなど
    """

    dt = config.dt
    te = config.duration
    tspan = np.arange(0.0, te + dt, dt)  # 0:dt:te

    # ==== プラント生成 (ここはあなたの Python クラスに合わせて修正) ====
    # cart = FurutaPendulum(init, measure_noise=0, sys_noise=0, dead_zone=0,
    #                       plant_param=[0.22,0.94, 9.1e-3, 0.0095, 0.0095,
    #                                    0.00017, 0.21,0.35, 0,0, 9.81, 3.61])
    # param = ACCESS_PLANT.get(cart, "plant_param")
    # param = cart.param

    # ---- 今は param をダミーにしておいて、後であなたの関数を呼ぶようにする ----
    param = None  # TODO: ここをあなたの Python 実装で置き換え

    # ==== 線形モデル Ac, Bc ====
    # Ac = Ac_FurutaPendulum(param)
    # Bc = Bc_FurutaPendulum(param)
    # 今はダミー 4次系 (実際にはあなたのAc, Bcを使う)
    Ac = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]], dtype=float)
    Bc = np.array([[0], [0], [0], [1]], dtype=float)

    Cc = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]], dtype=float)
    Dc = np.zeros((2, 1))

    # ==== 離散化 ====
    if config.time_mode == "discrete":
        Ad, Bd, Cd, Dd, _ = signal.cont2discrete((Ac, Bc, Cc, Dc), dt)
    else:
        Ad, Bd, Cd, Dd = None, None, None, None  # continuous のときは後で使用しない

    # ==== フィードバック・オブザーバーゲイン ====
    Q_lqr = np.diag([1.0, 100.0, 1.0, 1.0])
    R_lqr = np.array([[1.0]])

    if config.time_mode == "discrete":
        Fd = dlqr(Ad, Bd, Q_lqr, R_lqr)   # shape (1,4)
        Fod = dlqr(Ad.T, Cd.T,
                   np.diag([1.0, 1.0, 1.0, 1.0]),
                   np.array([[0.01]])).T  # オブザーバゲイン
        Fc = None
        Foc = None
    else:
        Fc = lqr(Ac, Bc, Q_lqr, R_lqr)
        Foc = lqr(Ac.T, Cc.T,
                  np.diag([1.0, 1.0, 1.0, 1.0]),
                  np.array([[0.01]])).T
        Fd = None
        Fod = None

    # ==== EKF 用の変数 (ここではほぼMATLABそのまま) ====
    if config.estimator == "EKF":
        P = np.eye(4)
        Qd = 1.0
        Rd = 0.01 * np.diag([0.02, 0.05])

    n_steps = len(tspan)
    T = np.zeros(n_steps)
    Y = np.zeros((n_steps, 2))
    Xhat = np.zeros((n_steps, 4))
    U = np.zeros(n_steps)

    # 初期出力 / 状態推定
    # ここでは「プラント = 線形モデル」として y = Cx としておく
    x_true = init.astype(float).copy()
    y = Cc @ x_true

    xh = np.array([y[0], y[1], 0.0, 0.0])  # 初期推定状態

    for i in range(n_steps):
        t = tspan[i]
        T[i] = t

        # ==== 観測 ====
        # 本当は cart.measure() を呼びたいところ
        # y = cart.measure
        y = Cc @ x_true  # 線形モデルの出力

        # ==== 状態推定 ====
        if i == 0:
            # すでに xh 初期化済み
            pass
        else:
            if config.time_mode == "discrete":
                if config.estimator == "EKF":
                    # 事前推定
                    xh_pre = Ad @ xh + Bd.flatten() * u
                    P_pre = Ad @ P @ Ad.T + Bd @ (Qd * Bd.T)
                    Gd = (P_pre @ Cd.T) @ np.linalg.inv(Cd @ P_pre @ Cd.T + Rd)
                    P = (np.eye(4) - Gd @ Cd) @ P_pre
                    xh = xh_pre + Gd @ (y - Cd @ xh_pre)
                else:
                    # 観測器
                    xh = Ad @ xh + Bd.flatten() * u - Fod @ (Cd @ xh - y)
            else:
                # 連続時間のオブザーバ (簡略版: オイラー積分)
                dxh = Ac @ xh + Bc.flatten() * u - Foc @ (Cc @ xh - y)
                xh = xh + dt * dxh

        # ==== フィードバック計算 ====
        if config.time_mode == "discrete":
            u = float(-Fd @ xh)
        else:
            u = float(-Fc @ xh)

        # ==== プラントを1ステップ進める ====
        # 本当は cart.apply_input(u, dt) を呼びたい
        # ここでは線形モデルをオイラー積分で進める
        dx_true = Ac @ x_true + Bc.flatten() * u
        x_true = x_true + dt * dx_true

        # ==== ログ ====
        Y[i, :] = y
        Xhat[i, :] = xh
        U[i] = u

    return SimResult(
        t=T.tolist(),
        y=Y.tolist(),
        xhat=Xhat.tolist(),
        u=U.tolist(),
    )
