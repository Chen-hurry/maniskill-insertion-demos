"""Replay buffer for offline and online recovery data."""

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    observation: Any
    action: Any
    reward: float
    next_observation: Any
    done: bool
    info: dict[str, Any]


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.storage: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def __len__(self) -> int:
        return len(self.storage)
