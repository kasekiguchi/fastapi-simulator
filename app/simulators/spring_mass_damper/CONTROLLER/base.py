from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...common.CONTROLLER.base import GenericController, ControllerParams as GenericControllerParams
from ..common.parameters import SmdParams
from ..common.matrices import smd_matrices


@dataclass
class ControllerParams(GenericControllerParams):
    """SMD controller settings (inherits generic params)."""


class CONTROLLER(GenericController):
    """Thin wrapper around generic controller for SMD."""

    def __init__(
        self,
        parameters: Optional[SmdParams] = None,
        settings: Optional[ControllerParams] = None,
        dt: Optional[float] = None,
    ) -> None:
        super().__init__(
            params=parameters or SmdParams(),
            matrices_fn=lambda p: smd_matrices(p),
            settings=settings,
            dt=dt,
        )
