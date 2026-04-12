from typing import Callable, Dict, Type
from .base import BaseSimulator
from .furuta_pendulum.simulator import FurutaPendulumSimulator
from .spring_mass_damper.simulator import SpringMassDamperSimulator
from .diff_drive import DiffDriveSimulator
from .tricycle.simulator import TricycleSimulator

SIM_REGISTRY: Dict[str, Callable[[], BaseSimulator]] = {
    "smd": lambda: SpringMassDamperSimulator(dt=0.01),
    "fp": lambda: FurutaPendulumSimulator(dt=0.01),
    "diffdrive": lambda: DiffDriveSimulator(dt=0.02),
    "tricycle": lambda: TricycleSimulator(dt=0.02),
}
