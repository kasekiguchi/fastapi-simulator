from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ...common.ESTIMATOR.base import GenericEstimator, EstimatorParams as GenericEstimatorParams
from ..common.Ac import Ac_furutaPendulum
from ..common.Bc import Bc_furutaPendulum
from ..common.physical_parameters import FurutaPendulumParams


@dataclass
class EstimatorParams(GenericEstimatorParams):
    """Furuta pendulum estimator settings (inherits generic params)."""


def _furuta_matrices(params: FurutaPendulumParams):
    A = Ac_furutaPendulum(params.as_array)
    B = Bc_furutaPendulum(params.as_array)
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    return A, B, C


class ESTIMATOR(GenericEstimator):
    """Thin wrapper around generic estimator for Furuta pendulum."""

    def __init__(
        self,
        parameters: Optional[FurutaPendulumParams] = None,
        settings: Optional[EstimatorParams] = None,
        dt: Optional[float] = None,
    ) -> None:
        super().__init__(
            params=parameters or FurutaPendulumParams(),
            matrices_fn=lambda p: _furuta_matrices(p),
            settings=settings,
            dt=dt,
        )
