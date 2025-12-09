from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...common.ESTIMATOR.base import GenericEstimator, EstimatorParams as GenericEstimatorParams
from ..common.parameters import SmdParams
from ..common.matrices import smd_matrices


@dataclass
class EstimatorParams(GenericEstimatorParams):
    """SMD estimator settings (inherits generic params)."""


class ESTIMATOR(GenericEstimator):
    """Thin wrapper around generic estimator for SMD."""

    def __init__(
        self,
        parameters: Optional[SmdParams] = None,
        settings: Optional[EstimatorParams] = None,
        dt: Optional[float] = None,
    ) -> None:
        super().__init__(
            params=parameters or SmdParams(),
            matrices_fn=lambda p: smd_matrices(p),
            settings=settings,
            dt=dt,
        )
