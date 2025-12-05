# app/runtime.py
from typing import Dict

from .simulators import SIM_REGISTRY
from .simulators.manager import SimulationManager


_managers: Dict[str, SimulationManager] = {}


def get_manager(sim_type: str) -> SimulationManager:
    """
    'smd' / 'fp' などの sim_type から SimulationManager を取得。
    なければ生成してキャッシュ。
    """
    key = sim_type.lower()
    if key in _managers:
        return _managers[key]

    if key not in SIM_REGISTRY:
        raise ValueError(f"Unknown sim_type: {sim_type}")

    sim = SIM_REGISTRY[key]()
    mgr = SimulationManager(sim)
    _managers[key] = mgr
    return mgr
