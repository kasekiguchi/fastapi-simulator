import numpy as np
from typing import Sequence, Callable, Tuple, Optional
from scipy import signal


def state_vector(state, expected_dim: int) -> np.ndarray:
    """Convert various state representations to a flat numpy vector."""
    try:
        if hasattr(state, "as_array"):
            arr = np.asarray(state.as_array, dtype=float).reshape(-1)
        else:
            arr = np.asarray(state, dtype=float).reshape(-1)
    except Exception:
        arr = np.zeros(expected_dim, dtype=float)

    if arr.size < expected_dim:
        arr = np.pad(arr, (0, expected_dim - arr.size))
    return arr


def to_square_matrix(values: Sequence[float], size: int) -> np.ndarray:
    """Create a square matrix from a flat list. Falls back to identity when invalid."""
    arr = np.asarray(values, dtype=float).flatten()
    if arr.size == size * size:
        return arr.reshape((size, size))
    if arr.size >= size:
        return np.diag(arr[:size])
    if arr.size == 1:
        return np.diag(np.repeat(arr[0], size))
    return np.eye(size)


def build_linear_model(
    matrices_fn: Callable[[], Tuple[np.ndarray, ...]],
    dt: float,
    time_mode: str,
    include_output: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Build continuous/discrete linear model given a callable that returns A,B,(C).
    If include_output is True, matrices_fn must return (A, B, C).
    Returns (A,B,Ad,Bd) or (A,B,C,Ad,Bd,Cd) depending on include_output.
    """
    mats = matrices_fn()
    if include_output:
        A, B, C = mats
    else:
        # Allow matrices_fn to return either (A,B) or (A,B,C); ignore extra outputs.
        if len(mats) >= 2:
            A, B = mats[0], mats[1]
        else:
            raise ValueError("matrices_fn must return at least (A, B)")
        C = np.eye(A.shape[0])

    if time_mode == "discrete":
        try:
            Ad, Bd, Cd, _, _ = signal.cont2discrete((A, B, C, np.zeros((C.shape[0], B.shape[1]))), dt)
        except Exception:
            Ad, Bd, Cd = A, B, C
    else:
        Ad, Bd, Cd = None, None, None

    if include_output:
        return A, B, C, Ad, Bd, Cd
    return A, B, Ad, Bd


def ackermann(A: np.ndarray, B: np.ndarray, poles: Sequence) -> np.ndarray:
    """Ackermann法による単入力系の極配置（重極対応）

    Returns:
        K: (nx,) ゲインベクトル。u = -K @ x で所望の極を配置。
    """
    n = A.shape[0]
    # 可制御性行列
    Mc = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    if abs(np.linalg.det(Mc)) < 1e-12:
        raise ValueError("System is not controllable")
    # 所望特性多項式の係数
    poly = np.array([1.0], dtype=complex)
    for p in poles:
        poly = np.convolve(poly, [1, -p])
    coeffs = np.real(poly)  # [1, a1, ..., an]
    # phi(A) = A^n + a1*A^(n-1) + ... + an*I
    phi_A = np.zeros_like(A, dtype=float)
    for i, c in enumerate(coeffs):
        phi_A += c * np.linalg.matrix_power(A, n - i)
    e_n = np.zeros(n)
    e_n[-1] = 1.0
    K = e_n @ np.linalg.inv(Mc) @ phi_A
    return K
