# app/simulators/manager.py
import asyncio
from typing import Callable, List, Optional

from .base import BaseSimulator, SimState

StateListener = Callable[[SimState], None]


class SimulationManager:
    """
    1つのシミュレータを dt 間隔で回す「制御装置」
    - start() / stop() でループを制御
    - dt_broadcastごとにリスナーへ状態を通知
    """

    def __init__(self, simulator: BaseSimulator, dt_broadcast: float = 1 / 30):
        self.sim = simulator
        self.dt_broadcast = dt_broadcast

        self._listeners: List[StateListener] = []
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def add_listener(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    async def set_params(self, **kwargs) -> None:
        async with self._lock:
            self.sim.set_params(**kwargs)

    async def apply_impulse(self, **kwargs) -> None:
        async with self._lock:
            self.sim.apply_impulse(**kwargs)

    async def _loop(self):
        acc = 0.0
        self._stop_event.clear()

        while not self._stop_event.is_set():
            async with self._lock:
                state = self.sim.step()

            acc += self.sim.dt
            if acc >= self.dt_broadcast:
                for listener in self._listeners:
                    try:
                        listener(state)
                    except Exception as e:
                        print("[SimulationManager] listener error:", e)
                acc = 0.0

            await asyncio.sleep(self.sim.dt)

    async def start(self):
        if self._task and not self._task.done():
            return  # 既に動いている
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task and not self._task.done():
            self._stop_event.set()
            await self._task
            self._task = None
