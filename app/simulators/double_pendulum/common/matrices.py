"""Double Pendulum線形化モデル（A, B 行列生成）

線形化点: カート上での両振り子直立状態
- x = vx = theta1 = theta2 = dtheta1 = dtheta2 = 0

状態ベクトル: x_vec = [x, vx, theta1, theta2, dtheta1, dtheta2]^T (6次)
入力ベクトル: u = [F]^T (1次)
"""

import numpy as np
from typing import Tuple
from .state import DoublePendulumParams


def double_pendulum_linearized_matrices(
    params: DoublePendulumParams,
    dt: float = 0.01,
    numerical_diff: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Double Pendulum線形化モデルを生成
    
    Returns:
        A: 連続系A行列 (6x6)
        B: 連続系B行列 (6x1)
        Ad: 離散系A行列 (6x6)
        Bd: 離散系B行列 (6x1)
    """
    
    if numerical_diff:
        return _linearize_numerical(params, dt)
    else:
        return _linearize_analytical(params, dt)


def _linearize_numerical(
    params: DoublePendulumParams,
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """数値微分によるヤコビアン計算"""
    
    delta = 1e-6
    n_state = 6
    n_input = 1
    
    A = np.zeros((n_state, n_state))
    B = np.zeros((n_state, n_input))
    
    x0 = np.zeros(n_state)
    u0 = np.zeros(n_input)
    
    # fx(x0, u0)を計算
    fx0 = _double_pendulum_dynamics(x0, u0, params)
    
    # A行列 = ∂f/∂x
    for j in range(n_state):
        x_perturb = x0.copy()
        x_perturb[j] += delta
        fx_perturb = _double_pendulum_dynamics(x_perturb, u0, params)
        A[:, j] = (fx_perturb - fx0) / delta
    
    # B行列 = ∂f/∂u
    for j in range(n_input):
        u_perturb = u0.copy()
        u_perturb[j] += delta
        fu_perturb = _double_pendulum_dynamics(x0, u_perturb, params)
        B[:, j] = (fu_perturb - fx0) / delta
    
    # 離散化
    Ad = np.eye(n_state) + A * dt
    Bd = B * dt
    
    return A, B, Ad, Bd


def _linearize_analytical(
    params: DoublePendulumParams,
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """簡略版解析的線形化"""
    
    n_state = 6
    n_input = 1
    
    # プレースホルダー: 簡略モデル
    A = np.zeros((n_state, n_state))
    B = np.zeros((n_state, n_input))
    
    # 位置-速度カップリング
    A[0, 1] = 1.0  # dx/dt = vx
    
    # 角速度カップリング
    A[2, 4] = 1.0  # dtheta1/dt = dtheta1
    A[3, 5] = 1.0  # dtheta2/dt = dtheta2
    
    # 復元力（簡略）
    m_total = params.M + params.m1 + params.m2
    A[1, 2] = params.m1 * params.lg1 * params.gravity / m_total
    A[1, 3] = params.m2 * params.lg2 * params.gravity / m_total
    
    A[4, 2] = -(params.M + params.m1 + params.m2) * params.gravity * params.lg1 / params.I1
    A[5, 3] = -(params.M + params.m1 + params.m2) * params.gravity * params.lg2 / params.I2
    
    # 入力
    B[1, 0] = 1.0 / m_total
    B[4, 0] = params.lg1 / params.I1
    B[5, 0] = params.lg2 / params.I2
    
    # 離散化
    Ad = np.eye(n_state) + A * dt
    Bd = B * dt
    
    return A, B, Ad, Bd


def _double_pendulum_dynamics(
    state: np.ndarray,
    control: np.ndarray,
    params: DoublePendulumParams
) -> np.ndarray:
    """
    Double Pendulum非線形ダイナミクス（Lagrange方程式から導出）
    
    状態: [x, vx, theta1, theta2, dtheta1, dtheta2]
    入力: [F]（カートへの水平力）
    
    戻り値: dx/dt (6次元)
    """
    
    x, vx, theta1, theta2, dtheta1, dtheta2 = state
    F = control[0]
    
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c12 = np.cos(theta1 - theta2)
    
    # 簡略版: 小角度近似
    # より詳細な導出については非線形方程式を参照
    
    # 質量行列の行列式
    M_total = params.M + params.m1 + params.m2
    det = params.I1 * params.I2 - params.I1 * params.m2 * params.L1**2 - params.I2 * params.m1 * params.L1**2
    
    # 状態方程式
    dx = np.zeros(6)
    
    # 位置の微分
    dx[0] = vx
    
    # 速度と角度の微分（簡略化）
    dx[1] = (F - params.friction * vx + 
             params.m1 * params.lg1 * params.gravity * s1 + 
             params.m2 * params.lg2 * params.gravity * s2) / M_total
    
    dx[2] = dtheta1
    dx[3] = dtheta2
    
    # 角加速度（Lagrange方程式の簡略化）
    dx[4] = -(params.M + params.m1 + params.m2) * params.gravity * params.lg1 * s1 / params.I1
    dx[5] = -(params.M + params.m1 + params.m2) * params.gravity * params.lg2 * s2 / params.I2
    
    return dx
