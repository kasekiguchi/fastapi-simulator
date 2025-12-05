
import numpy as np
from scipy import linalg

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

