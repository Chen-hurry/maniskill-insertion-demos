"""Hybrid recovery controller that fuses multiple risk signals."""

from typing import Any


class HybridRecovery:
    def __init__(self, detectors: list, recovery_actor, rule_based=None, threshold: float = 0.5):
        self.detectors = detectors
        self.recovery_actor = recovery_actor
        self.rule_based = rule_based
        self.threshold = threshold

    def score(self, observation: Any, action: Any | None = None) -> float:
        if not self.detectors:
            return 0.0
        scores = []
        for detector in self.detectors:
            if hasattr(detector, "score"):
                scores.append(float(detector.score(observation, action)))
        return sum(scores) / max(len(scores), 1)

    def act(self, observation: Any, nominal_action: Any) -> Any:
        if self.score(observation, nominal_action) >= self.threshold:
            return self.recovery_actor.act(observation)
        return nominal_action
