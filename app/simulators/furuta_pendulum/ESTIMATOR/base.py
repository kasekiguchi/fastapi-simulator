from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal

from ..FURUTA_PENDULUM.base  import FurutaPendulumParams
from ..common.Ac import Ac_furutaPendulum
from ..common.Bc import Bc_furutaPendulum

from ...utils.linear_optimal import lqr, dlqr

from ..common.state import FurutaPendulumState
from ..common.physical_parameters import FurutaPendulumParams

@dataclass
class ESTIMATOR():
    """
    """

    def __init__(
        self,
        parameters,
        settings
    ) -> None:      
        self.P = parameters or FurutaPendulumParams()
        # 線形モデル
        Ac = Ac_furutaPendulum(params=self.P.as_array)
        Bc = Bc_furutaPendulum(params=self.P.as_array)
        Cc = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        Dc = np.zeros((2, 1))
        self.state = FurutaPendulumState()
        self.xh = self.state.as_array
        self.params = settings
        if self.params.time_mode == "discrete":
                  self.A, self.B, self.C, self.D, _ = signal.cont2discrete((self.Ac, self.Bc, self.Cc, self.Dc), self.params.dt)
        else:
            self.A = Ac
            self.B = Bc
            self.C = Cc
            self.D = Dc
            
        Q = np.diag([1.0, 100.0, 1.0, 1.0])
        R = np.diag([1.0, 1.0])

        if self.params.time_mode == "discrete":
            self.Fo = dlqr(self.A.T,self.C.T,Q,R).T
        else:
            self.Fo = lqr(self.A.T,self.C.T,Q,R).T

                    
    def estimate(self,u,y)->None:
        # 推定
        if self.params.time_mode == "discrete":
            if self.params.estimator == "EKF":
                xh_pre = self.A @ self.xh + self.B.flatten() * u
                P_pre = self.A @ self.P @ self.A.T + self.B @ (self.Q * self.B.T)
                self.Gd = (P_pre @ self.C.T) @ np.linalg.inv(self.C @ P_pre @ self.C.T + self.R)
                self.P = (np.eye(4) - self.G @ self.C) @ P_pre
                self.xh = xh_pre + self.G @ (y - self.C @ xh_pre)
            else:
                self.xh = self.A @ self.xh + self.B.flatten() * u - self.self.Fo @ (self.C @ self.xh - y)
        else:
            dxh = self.A @ self.xh + self.B.flatten() * u - self.Fo @ (self.C @ self.xh - y)
            self.xh = self.xh + self.params.dt * dxh
        return self.state.set(self.xh)

