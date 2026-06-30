"""Risk and value critic interfaces."""

from typing import Any


class Critic:
    def value(self, observation: Any, action: Any | None = None) -> float:
        return 0.0

    def risk(self, observation: Any, action: Any | None = None) -> float:
        return 1.0 - self.value(observation, action)
