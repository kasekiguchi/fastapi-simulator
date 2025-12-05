# app/simulators/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SimState:
    """すべてのシミュレータが共通で持つ最小状態（時間だけ）"""
    t: float = 0.0


class BaseSimulator(ABC):
    """すべてのシミュレータの共通インターフェース"""
    dt: float

    @abstractmethod
    def reset(self) -> None:
        """状態を初期化"""
        ...

    @abstractmethod
    def step(self) -> SimState:
        """1ステップ進めて現在の状態を返す"""
        ...

    @abstractmethod
    def apply_impulse(self, **kwargs) -> None:
        """クリックなどで一時的な外力/入力を与える"""
        ...

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """物理パラメータを変更"""
        ...
