from __future__ import annotations

from typing import Any

import numpy as np

from .base import _BaseControllerStrategy


class PIDController(_BaseControllerStrategy):
    """Simple PID on position: u = -(Kp*(p-ref) + Ki*∫(p-ref) + Kd*v)."""

    def __init__(
        self,
        kp: float = 0.0,
        ki: float = 0.0,
        kd: float = 0.0,
        dt: float = 0.01,
        ref: float = 0.0,
        pos_index: int = 0,
        vel_index: int | None = 1,
        ref_pos_index: int | None = None,
        ref_vel_index: int | None = None,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.ref = 0.0  # set via set_reference
        self.pos_index = int(pos_index)
        self.vel_index = None if vel_index is None else int(vel_index)
        # Default ref indices follow state indices if not provided.
        self.ref_pos_index = int(ref_pos_index) if ref_pos_index is not None else self.pos_index
        self.ref_vel_index = None if ref_vel_index is None else int(ref_vel_index)
        self.ref_provider = None  # callable with sample(t) or __call__(t)
        self._int_err = 0.0
        # apply initial reference if provided
        if ref is not None:
            self.set_reference(ref)

    def reset(self) -> None:
        self._int_err = 0.0

    def set_params(self, params: Any) -> None:
        # No dependency on plant params, but keep signature for compatibility.
        return

    def set_gains(self, kp: float | None = None, ki: float | None = None, kd: float | None = None) -> None:
        if kp is not None:
            self.kp = float(kp)
        if ki is not None:
            self.ki = float(ki)
        if kd is not None:
            self.kd = float(kd)

    def set_reference(self, ref: float | dict | None) -> None:
        """Accept scalar, sequence, dict, or reference provider (has sample or is callable)."""
        if ref is None:
            return
        # Reference provider (e.g., Reference class with sample(t))
        if hasattr(ref, "sample") and callable(ref.sample):
            self.ref_provider = ref.sample
            return
        if callable(ref):
            self.ref_provider = ref
            return

        # Static reference
        if isinstance(ref, dict):
            val = ref.get("p")
            if val is None:
                val = ref.get("pos", ref.get("position"))
            if val is None and self.ref_pos_index is not None:
                try:
                    seq_val = ref.get("values")
                    if seq_val is not None:
                        val = seq_val[self.ref_pos_index]
                except Exception:
                    pass
        elif isinstance(ref, (list, tuple)):
            val = None
            try:
                val = ref[self.ref_pos_index]
            except Exception:
                pass
        else:
            val = ref
        try:
            self.ref = float(val)
        except (TypeError, ValueError):
            pass

    def set_indices(
        self,
        pos_index: int | None = None,
        vel_index: int | None = None,
        ref_pos_index: int | None = None,
        ref_vel_index: int | None = None,
    ) -> None:
        if pos_index is not None:
            self.pos_index = int(pos_index)
        if vel_index is not None:
            self.vel_index = None if vel_index is None else int(vel_index)
        if ref_pos_index is not None:
            self.ref_pos_index = None if ref_pos_index is None else int(ref_pos_index)
        if ref_vel_index is not None:
            self.ref_vel_index = None if ref_vel_index is None else int(ref_vel_index)

    def _unwrap_state(self, state) -> tuple[float, float]:
        try:
            if hasattr(state, "as_array"):
                arr = np.asarray(state.as_array, dtype=float).flatten()
            else:
                arr = np.asarray(state, dtype=float).flatten()
        except Exception:
            arr = np.zeros(2, dtype=float)
        if self.pos_index is None or self.pos_index >= arr.size:
            arr = np.pad(arr, (0, self.pos_index + 1 - arr.size)) if self.pos_index is not None else arr
        pos = float(arr[self.pos_index]) if self.pos_index is not None else 0.0

        if self.vel_index is None:
            vel = 0.0
        else:
            if self.vel_index >= arr.size:
                arr = np.pad(arr, (0, self.vel_index + 1 - arr.size))
            vel = float(arr[self.vel_index])
        return pos, vel

    def _to_array(self, sample) -> np.ndarray:
        if sample is None:
            return np.zeros(1, dtype=float)
        if hasattr(sample, "as_array"):
            try:
                return np.asarray(sample.as_array, dtype=float).flatten()
            except Exception:
                pass
        if hasattr(sample, "pos"):
            try:
                pos = np.asarray(sample.pos, dtype=float).flatten()
                others = []
                for attr in ("theta", "v", "omega"):
                    if hasattr(sample, attr):
                        others.append(float(getattr(sample, attr)))
                return np.concatenate([pos, np.asarray(others, dtype=float)])
            except Exception:
                pass
        try:
            return np.asarray(sample, dtype=float).flatten()
        except Exception:
            return np.zeros(1, dtype=float)

    def _get_reference(self, t: float | None) -> tuple[float, float]:
        if self.ref_provider:
            try:
                sample = self.ref_provider(t if t is not None else 0.0)
            except Exception:
                sample = None
            arr = self._to_array(sample)
            ref_pos_idx = self.ref_pos_index if self.ref_pos_index is not None else 0
            ref_vel_idx = self.ref_vel_index
            if ref_pos_idx >= arr.size:
                arr = np.pad(arr, (0, ref_pos_idx + 1 - arr.size))
            ref_p = float(arr[ref_pos_idx])
            if ref_vel_idx is None:
                ref_v = 0.0
            else:
                if ref_vel_idx >= arr.size:
                    arr = np.pad(arr, (0, ref_vel_idx + 1 - arr.size))
                ref_v = float(arr[ref_vel_idx])
            return ref_p, ref_v
        return self.ref, 0.0

    def compute(self, state) -> float:
        p, v = self._unwrap_state(state)
        t = getattr(state, "t", None)
        ref_p, ref_v = self._get_reference(t)
        err = p - ref_p
        vel_err = v - ref_v
        self._int_err += err * self.dt
        u = -(self.kp * err + self.ki * self._int_err + self.kd * vel_err)
        return float(u)
