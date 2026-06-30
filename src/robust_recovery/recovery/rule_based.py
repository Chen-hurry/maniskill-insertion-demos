"""Simple rule-based recovery primitives."""

from typing import Any


class RuleBasedRecovery:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def recover(self, observation: Any) -> dict[str, Any]:
        return {"action": "reset_or_retract", "attempts": self.max_attempts}
