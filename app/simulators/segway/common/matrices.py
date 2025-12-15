"""Segway線形化モデル（A, B 行列生成）

線形化点: 倒立平衡状態
- x = vx = y = vy = phi = psi = dphi = dtheta = dpsi = 0
- theta = 0 (倒立)

状態ベクトル: x_vec = [x, vx, y, vy, phi, theta, psi, dphi, dtheta, dpsi]^T (10次)
入力ベクトル: u = [tau_left, tau_right]^T (2次)
"""

import numpy as np
from typing import Tuple
from .state import SegwayParams


def segway_linearized_matrices(
    params: SegwayParams,
    dt: float = 0.01,
    numerical_diff: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Segway線形化モデルを生成
    
    Returns:
        A: 連続系A行列 (10x10)
        B: 連続系B行列 (10x2)
        Ad: 離散系A行列 (10x10)
        Bd: 離散系B行列 (10x2)
    """
    
    if numerical_diff:
        # 数値微分による線形化
        return _linearize_numerical(params, dt)
    else:
        # 解析的線形化（簡略版）
        return _linearize_analytical(params, dt)


def _linearize_numerical(
    params: SegwayParams,
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """数値微分によるヤコビアン計算"""
    
    delta = 1e-6
    n_state = 10
    n_input = 2
    
    A = np.zeros((n_state, n_state))
    B = np.zeros((n_state, n_input))
    
    x0 = np.zeros(n_state)
    u0 = np.zeros(n_input)
    
    # fx(x0, u0)を計算
    fx0 = _segway_dynamics(x0, u0, params)
    
    # A行列 = ∂f/∂x
    for j in range(n_state):
        x_perturb = x0.copy()
        x_perturb[j] += delta
        fx_perturb = _segway_dynamics(x_perturb, u0, params)
        A[:, j] = (fx_perturb - fx0) / delta
    
    # B行列 = ∂f/∂u
    for j in range(n_input):
        u_perturb = u0.copy()
        u_perturb[j] += delta
        fu_perturb = _segway_dynamics(x0, u_perturb, params)
        B[:, j] = (fu_perturb - fx0) / delta
    
    # 離散化: Ad = I + A*dt, Bd = B*dt (1次のEuler法)
    Ad = np.eye(n_state) + A * dt
    Bd = B * dt
    
    return A, B, Ad, Bd


def _linearize_analytical(
    params: SegwayParams,
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    簡略版解析的線形化（簡易モデル）
    実際にはより詳細な導出が必要
    """
    # プレースホルダー実装
    n_state = 10
    n_input = 2
    
    # 簡略版: 水平移動とピッチのカップリング
    A = np.eye(n_state) * (-0.1)
    B = np.zeros((n_state, n_input))
    
    # 位置速度関係
    A[0, 1] = 1.0  # dx/dt = vx
    A[2, 3] = 1.0  # dy/dt = vy
    
    # ピッチダイナミクス（簡略）
    A[5, 9] = 1.0  # dtheta/dt = dtheta
    A[9, 5] = -params.gravity / params.h  # d²theta/dt² ~ -g/h * theta
    
    # 入力による加速
    B[1, :] = 1.0 / (params.M + params.m) / params.r
    B[3, :] = 1.0 / (params.M + params.m) / params.r
    
    # 離散化
    Ad = np.eye(n_state) + A * dt
    Bd = B * dt
    
    return A, B, Ad, Bd


def _segway_dynamics(
    state: np.ndarray,
    control: np.ndarray,
    params: SegwayParams
) -> np.ndarray:
    """
    Segway非線形ダイナミクス
    
    状態: [x, vx, y, vy, phi, theta, psi, dphi, dtheta, dpsi]
    入力: [tau_left, tau_right]
    
    戻り値: dx/dt (10次元)
    """
    
    x, vx, y, vy, phi, theta, psi, dphi, dtheta, dpsi = state
    tau_left, tau_right = control
    
    # 状態方程式の微分項
    dx = np.zeros(10)
    
    # 位置の微分（並進）
    dx[0] = vx  # dx/dt = vx
    dx[2] = vy  # dy/dt = vy
    
    # ボディ方向の速度更新
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    
    # 小角度近似を使用（ロール角 phi が小さい）
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 車輪駆動からの力（簡略モデル）
    tau_total = tau_left + tau_right
    tau_yaw = (tau_right - tau_left) * params.L / 2.0
    
    # 直進力（ボディ座標系）
    F_forward = tau_total / params.r
    
    # 加速度（ダンピング項を含む）
    total_mass = params.M + params.m
    dx[1] = (F_forward * cos_theta - params.Damp * vx) / total_mass
    dx[3] = (F_forward * sin_theta - params.Damp * vy) / total_mass
    
    # 角度の微分
    dx[4] = dphi  # dphi/dt = dphi
    dx[5] = dtheta  # dtheta/dt = dtheta
    dx[6] = dpsi  # dpsi/dt = dpsi
    
    # 角加速度（簡略モデル）
    # ピッチ角ダイナミクス（倒立復元力）
    pitch_torque = (params.M + params.m) * params.gravity * params.h * sin_theta
    dx[9] = (pitch_torque - params.Damp * dpsi) / params.Izz
    
    # ロール角ダイナミクス（重力復元）
    roll_torque = (params.M + params.m) * params.gravity * params.h * sin_phi
    dx[7] = (roll_torque - params.Damp * dphi) / params.Ixx
    
    # ヨー角ダイナミクス
    dx[8] = (tau_yaw - params.Damp * dpsi) / params.Izz
    
    return dx
