"""Vision-language-action policy adapter."""

from typing import Any

from robust_recovery.policies.base_actor import BaseActor


class VLAPolicy(BaseActor):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def act(self, observation: Any, deterministic: bool = False) -> Any:
        return {"model": self.model_name, "action": "placeholder"}
