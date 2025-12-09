from __future__ import annotations

import numpy as np

from .base import _BaseControllerStrategy
from ..linear_utils import state_vector


class StateFeedbackController(_BaseControllerStrategy):
    def __init__(self, gain):
        self.gain = np.asarray(gain, dtype=float).flatten()

    def set_params(self, params) -> None:
        return

    def compute(self, state):
        x = state_vector(state, expected_dim=self.gain.size or 1)
        try:
            return -float(np.dot(self.gain, x[: self.gain.size]))
        except Exception:
            return 0.0
