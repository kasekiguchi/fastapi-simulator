# app/runtime.py
from .simulators import SIM_REGISTRY
from .simulators.manager import SimulationManager


def create_manager(sim_type: str) -> SimulationManager:
    """
    sim_type から新しい SimulationManager を生成する。
    接続ごとに独立したシミュレータを持つため、キャッシュはしない。
    """
    key = sim_type.lower()
    if key not in SIM_REGISTRY:
        raise ValueError(f"Unknown sim_type: {sim_type}")
    sim = SIM_REGISTRY[key]()
    return SimulationManager(sim)
