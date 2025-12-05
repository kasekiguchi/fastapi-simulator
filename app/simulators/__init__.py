from typing import Callable, Dict, Type
from .base import BaseSimulator
from .SMD.simulator import SmdSimulator
from .furuta_pendulum.simulator import FurutaPendulumSimulator

SIM_REGISTRY: Dict[str, Callable[[], BaseSimulator]] = {
    "smd": lambda: SmdSimulator(dt=0.01),
    "fp": lambda: FurutaPendulumSimulator(dt=0.01),
}
