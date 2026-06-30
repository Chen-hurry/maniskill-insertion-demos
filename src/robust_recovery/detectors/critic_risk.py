"""Critic-derived risk detector."""

from typing import Any


class CriticRiskDetector:
    def __init__(self, critic, threshold: float = 0.45):
        self.critic = critic
        self.threshold = threshold

    def score(self, observation: Any, action: Any | None = None) -> float:
        return float(self.critic.risk(observation, action))

    def is_failure_risk(self, observation: Any, action: Any | None = None) -> bool:
        return self.score(observation, action) >= self.threshold
