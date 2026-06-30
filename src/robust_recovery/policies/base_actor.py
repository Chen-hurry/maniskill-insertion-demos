"""Base action policy interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseActor(ABC):
    @abstractmethod
    def act(self, observation: Any, deterministic: bool = False) -> Any:
        """Return an action for one observation."""
