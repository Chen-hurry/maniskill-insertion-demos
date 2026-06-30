"""Anomaly injection utilities."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AnomalySpec:
    name: str
    probability: float = 1.0
    severity: float = 0.5


class AnomalyInjector:
    def __init__(self, specs: list[AnomalySpec]):
        self.specs = specs

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        updated = dict(state)
        updated.setdefault("active_anomalies", [])
        for spec in self.specs:
            updated["active_anomalies"].append(
                {"name": spec.name, "probability": spec.probability, "severity": spec.severity}
            )
        return updated
