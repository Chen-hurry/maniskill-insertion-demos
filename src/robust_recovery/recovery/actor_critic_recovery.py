"""Actor-critic recovery controller."""

from typing import Any


class ActorCriticRecovery:
    def __init__(self, detector, recovery_actor):
        self.detector = detector
        self.recovery_actor = recovery_actor

    def act(self, observation: Any, nominal_action: Any) -> Any:
        if self.detector.is_failure_risk(observation, nominal_action):
            return self.recovery_actor.act(observation)
        return nominal_action
