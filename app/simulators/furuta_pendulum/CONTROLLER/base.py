# Path: app/furuta_pendulum/FURUTA_PENDULUM/base.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal

from ..FURUTA_PENDULUM.base  import FurutaPendulumParams
from app.simulators.furuta_pendulum.common.Ac import Ac_furutaPendulum
from app.simulators.furuta_pendulum.common.Bc import Bc_furutaPendulum
from .calc_input import calc_input
from ...utils.linear_optimal import lqr, dlqr

from ..common.physical_parameters import FurutaPendulumParams

@dataclass
class ControllerParams:
    Ac: np.array
    Bc: np.array
    Fc: np.array
    time_mode = "discrete"    
    Q: np.array
    R: np.array
    dt = 0.01

@dataclass
# class ControllerState(SimState):
#    """ for dynamic filter """
#     theta: float = 0.0
#     dtheta: float = 0.0
#     phi: float = 0.0
#     dphi: float = 0.0
#     @property
#     def as_array(self) -> np.ndarray:
#         return np.array(
#             [
#                 self.theta,
#                 self.phi,
#                 self.dtheta,
#                 self.dphi
#             ])

class CONTROLLER():
    """
    """

    def __init__(
        self,
        parameters,
        settings
    ) -> None:
        self.P = parameters or FurutaPendulumParams()
        self.params = settings 
        self.input = 0.0
            
        # 線形モデル
        Ac = Ac_furutaPendulum(params=self.P.as_array)
        Bc = Bc_furutaPendulum(params=self.P.as_array)
        Cc = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        Dc = np.zeros((2, 1))

        if self.params.time_mode == "discrete":
            self.A, self.B, self.C, self.D, _ = signal.cont2discrete((self.Ac, self.Bc, self.Cc, self.Dc), self.params.dt)
        else:
            self.A = Ac
            self.B = Bc
            self.C = Cc
            self.D = Dc
        

        Q_lqr = np.diag([1.0, 100.0, 1.0, 1.0])
        R_lqr = np.array([[1.0]])

        if self.params.time_mode == "discrete":
            self.F = dlqr(self.A, self.B, Q_lqr, R_lqr)
            self.Fo = dlqr(
                self.A.T,
                self.C.T,
                np.diag([1.0, 1.0, 1.0, 1.0]),
                np.diag([0.01,0.01]),
            ).T
        else:
            self.F = lqr(self.A, self.B, Q_lqr, R_lqr)
            self.Fo = lqr(
                self.A.T,
                self.C.T,
                np.diag([1.0, 1.0, 1.0, 1.0]),
                np.diag([0.01,0.01]),
            ).T


    # ---- MATLABと同名のメソッドを外部ファイルからバインド ----
    Ac_func = Ac_furutaPendulum
    Bc_func = Bc_furutaPendulum
    calc_input = calc_input
        
    def reset(self) -> None:
        self.input = 0.0

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, float(v))
        self.Ac = self.Ac_func(self.params)
        self.Bc = self.Bc_func(self.params)

