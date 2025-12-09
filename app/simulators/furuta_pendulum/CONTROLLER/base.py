from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...common.CONTROLLER.base import GenericController, ControllerParams as GenericControllerParams
from ..common.Ac import Ac_furutaPendulum
from ..common.Bc import Bc_furutaPendulum
from ..common.physical_parameters import FurutaPendulumParams


@dataclass
class ControllerParams(GenericControllerParams):
    """Furuta pendulum controller settings (inherits generic params)."""


def _furuta_matrices(params: FurutaPendulumParams):
    A = Ac_furutaPendulum(params.as_array)
    B = Bc_furutaPendulum(params.as_array)
    return A, B


class CONTROLLER(GenericController):
    """Thin wrapper around generic controller for Furuta pendulum."""

    def __init__(
        self,
        parameters: Optional[FurutaPendulumParams] = None,
        settings: Optional[ControllerParams] = None,
        dt: Optional[float] = None,
    ) -> None:
        super().__init__(
            params=parameters or FurutaPendulumParams(),
            matrices_fn=lambda p: _furuta_matrices(p),
            settings=settings,
            dt=dt,
        )
