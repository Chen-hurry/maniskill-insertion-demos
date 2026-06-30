"""Visual anomaly detector interface."""

from typing import Any


class VisualAnomalyDetector:
    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def score(self, observation: Any) -> float:
        return 0.0

    def is_anomaly(self, observation: Any) -> bool:
        return self.score(observation) >= self.threshold
