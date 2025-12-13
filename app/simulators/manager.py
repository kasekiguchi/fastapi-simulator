# app/simulators/manager.py
import asyncio
from typing import Callable, List, Optional, Awaitable, Union

from .base import BaseSimulator, SimState

StateListener = Callable[[SimState], None]
StopListener = Callable[[bool, Optional[SimState]], Union[Awaitable[None], None]]


class SimulationManager:
    """
    1つのシミュレータを dt 間隔で回す「制御装置」
    - start() / stop() でループを制御
    - dt_broadcastごとにリスナーへ状態を通知
    """

    DEFAULT_DURATION = 20.0

    def __init__(self, simulator: BaseSimulator, dt_broadcast: float = 1 / 30):
        self.sim = simulator
        self.dt_broadcast = dt_broadcast

        self._listeners: List[StateListener] = []
        self._stop_listeners: List[StopListener] = []
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._duration_limit: Optional[float] = self.DEFAULT_DURATION
        self._start_time: float = 0.0

    def add_listener(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    def add_stop_listener(self, listener: StopListener) -> None:
        self._stop_listeners.append(listener)

    async def set_params(self, **kwargs) -> None:
        async with self._lock:
            self.sim.set_params(**kwargs)

    async def apply_impulse(self, **kwargs) -> None:
        async with self._lock:
            self.sim.apply_impulse(**kwargs)

    async def set_initial(self, **kwargs) -> None:
        if hasattr(self.sim, "set_initial"):
            async with self._lock:
                self.sim.set_initial(**kwargs)

    async def reset(self) -> None:
        async with self._lock:
            self.sim.reset()
            # After reset, broadcast t=0 state if available
            state = None
            getter = getattr(self.sim, "get_public_state", None)
            if callable(getter):
                try:
                    state = getter()
                except Exception:
                    state = None
            if state:
                for listener in self._listeners:
                    try:
                        listener(state)
                    except Exception as e:
                        print("[SimulationManager] listener error on reset:", e)

    async def set_control_params(self, control_params=None) -> None:
        if hasattr(self.sim, "set_control_params"):
            async with self._lock:
                self.sim.set_control_params(control_params=control_params)
    async def set_estimator_params(self, estimator_params=None) -> None:
        if hasattr(self.sim, "set_estimator_params"):
            async with self._lock:
                self.sim.set_estimator_params(estimator_params=estimator_params)

    async def set_reference(self, reference=None) -> None:
        if hasattr(self.sim, "set_reference"):
            async with self._lock:
                self.sim.set_reference(reference)

    async def set_exp_mode(self, exp_mode: Optional[bool]) -> None:
        if exp_mode is None:
            return
        if hasattr(self.sim, "set_exp_mode"):
            async with self._lock:
                self.sim.set_exp_mode(bool(exp_mode))

    async def get_trace(self):
        """Return aggregated simulation trace if available."""
        getter = getattr(self.sim, "get_trace", None)
        if not callable(getter):
            return None
        async with self._lock:
            try:
                return getter()
            except Exception:
                return None

    async def _loop(self):
        acc = 0.0
        self._stop_event.clear()
        state: Optional[SimState] = None
        auto_stopped = False
        try:
            while not self._stop_event.is_set():
                async with self._lock:
                    try:
                        state = self.sim.step()
                    except Exception as e:
                        print("[SimulationManager] step error:", e)
                        state = None

                acc += self.sim.dt
                if acc >= self.dt_broadcast and state is not None:
                    for listener in self._listeners:
                        try:
                            listener(state)
                        except Exception as e:
                            print("[SimulationManager] listener error:", e)
                    acc = 0.0

                if self._duration_limit is not None:
                    t_now = self._current_time(state)
                    if (t_now - self._start_time) >= self._duration_limit:
                        auto_stopped = True
                        self._stop_event.set()
                        break

                await asyncio.sleep(self.sim.dt)
        except Exception as loop_err:
            print("[SimulationManager] loop abort:", loop_err)
            raise
        finally:
            if auto_stopped:
                print(f"[SimulationManager] auto-stopped at t={self._current_time(state):.3f}")
            await self._notify_stop(auto_stopped, state)
            self._task = None

    def _current_time(self, state: Optional[SimState]) -> float:
        """Return current simulation time, tolerant to missing state."""
        if state is not None and hasattr(state, "t"):
            try:
                return float(state.t)
            except Exception:
                pass
        sim_state = getattr(self.sim, "state", None)
        if sim_state is not None and hasattr(sim_state, "t"):
            try:
                return float(sim_state.t)
            except Exception:
                pass
        return self._start_time

    async def start(self, duration: Optional[float] = None):
        if self._task and not self._task.done():
            return  # 既に動いている
        self._duration_limit = self._normalize_duration(duration)
        self._start_time = self._current_time(getattr(self.sim, "state", None))
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task and not self._task.done():
            self._stop_event.set()
            await self._task
            self._task = None

    def _normalize_duration(self, duration: Optional[float]) -> Optional[float]:
        """Normalize duration (<=0 -> no limit)."""
        val = duration if duration is not None else self.DEFAULT_DURATION
        if val is None:
            return None
        try:
            v = float(val)
        except (TypeError, ValueError):
            return self.DEFAULT_DURATION
        return v if v > 0 else None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _notify_stop(self, auto: bool, state: Optional[SimState]) -> None:
        """Inform listeners that the loop stopped."""
        for listener in list(self._stop_listeners):
            try:
                result = listener(auto, state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print("[SimulationManager] stop listener error:", e)

    def current_limit(self) -> Optional[float]:
        """Return currently configured duration limit (None if unlimited)."""
        return self._duration_limit
