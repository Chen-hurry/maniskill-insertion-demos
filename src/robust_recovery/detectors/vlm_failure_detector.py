"""VLM-based failure detector placeholder."""

from typing import Any


class VLMFailureDetector:
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

    def analyze(self, observation: Any, instruction: str = "") -> dict[str, Any]:
        return {"failure": False, "confidence": 0.0, "instruction": instruction}

    def is_failure(self, observation: Any, instruction: str = "") -> bool:
        result = self.analyze(observation, instruction)
        return bool(result["failure"]) and result["confidence"] >= self.confidence_threshold
