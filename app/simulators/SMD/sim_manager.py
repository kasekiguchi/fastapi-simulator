# backend/sim/sim_manager.py
import asyncio
from typing import Callable, List, Awaitable, Optional
from .simulator import Simulator, PhysicsParams, SimState

StateListener = Callable[[SimState], None]

class SimulationManager:
    """
    - 内部で Simulator を保持
    - dt_sim 間隔で step()
    - dt_broadcast ごとにリスナーへ状態通知
    """

    def __init__(self, dt_sim: float = 0.01, dt_broadcast: float = 1 / 30):
        self.sim = Simulator(dt=dt_sim)
        self.dt_broadcast = dt_broadcast

        self._listeners: List[StateListener] = []
        self._lock = asyncio.Lock()
        self._running = False

    def add_listener(self, listener: StateListener) -> None:
        """状態送信用リスナーを登録（WebSocket 側で使用）"""
        self._listeners.append(listener)

    async def set_params(self, params: PhysicsParams) -> None:
        """パラメータを更新（REST or WebSocket から呼ばれる）"""
        async with self._lock:
            self.sim.set_params(params)

    async def apply_impulse(self, force: float) -> None:
        """外力を加える（例えばマウスクリック時）"""
        async with self._lock:
            self.sim.apply_impulse(force)

    async def run_forever(self) -> None:
        """
        常時ループ
        - dt_sim で step()
        - dt_broadcast ごとにリスナーへ状態通知
        """
        if self._running:
            return
        self._running = True

        acc = 0.0
        while True:
            async with self._lock:
                state = self.sim.step()

            acc += self.sim.dt

            if acc >= self.dt_broadcast:
                # 状態を全リスナーへ通知
                for listener in self._listeners:
                    try:
                        listener(state)
                    except Exception as e:
                        # listener 内部のエラーで止まらないように
                        print("[SimulationManager] listener error:", e)
                acc = 0.0

            # 単純に dt_sim だけ sleep
            await asyncio.sleep(self.sim.dt)
