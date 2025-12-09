from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy import linalg

from .base import _LinearControllerStrategy
from ..linear_utils import state_vector, to_square_matrix


class LQRController(_LinearControllerStrategy):
    def __init__(self, params, dt: float, time_mode: str, matrices_fn: Callable[[Any], tuple], q: Sequence[float], r: Sequence[float]):
        self.q = q
        self.r = r
        super().__init__(params, dt, time_mode, matrices_fn)
        size_x = self.Ad.shape[0] if self.Ad is not None else self.nx
        size_u = self.Bd.shape[1] if self.Bd is not None else self.nu
        self.Q = to_square_matrix(q, size_x)
        self.R = to_square_matrix(r, size_u)
        self.K = np.zeros((size_u, size_x))
        self._design_gain()

    def _design_gain(self):
        if self.time_mode == "discrete" and self.Ad is not None and self.Bd is not None:
            try:
                P = linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
                BtP = self.Bd.T @ P
                self.K = np.linalg.inv(BtP @ self.Bd + self.R) @ (BtP @ self.Ad)
                return
            except Exception:
                pass
        try:
            P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            BtP = self.B.T @ P
            self.K = np.linalg.inv(BtP @ self.B + self.R) @ (BtP @ self.A)
            print(f"[LQR] discrete gain matrix (): {self.K}")

        except Exception:
            self.K = np.zeros((self.nu, self.nx))
        print(f"[LQR] discrete gain matrix (KNV0): {self.K}")


    def set_params(self, params) -> None:
        super().set_params(params)
        size_x = self.Ad.shape[0] if self.Ad is not None else self.nx
        size_u = self.Bd.shape[1] if self.Bd is not None else self.nu
        self.Q = to_square_matrix(self.q, size_x)
        self.R = to_square_matrix(self.r, size_u)
        self.K = np.zeros((size_u, size_x))
        self._design_gain()

    def compute(self, state):
        x = state_vector(state, expected_dim=self.nx)
        try:
            return -float(self.K @ x[: self.nx])
        except Exception:
            return 0.0
