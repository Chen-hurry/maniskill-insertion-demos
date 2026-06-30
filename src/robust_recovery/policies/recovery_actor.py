"""Learned recovery policy placeholder."""

from typing import Any

from robust_recovery.policies.base_actor import BaseActor


class RecoveryActor(BaseActor):
    def act(self, observation: Any, deterministic: bool = False) -> Any:
        return observation.get("fallback_action", 0) if isinstance(observation, dict) else 0
