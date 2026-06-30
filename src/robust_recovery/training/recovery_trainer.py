"""Trainer for recovery actor and recovery-trigger modules."""

from dataclasses import dataclass


@dataclass
class RecoveryTrainerConfig:
    total_steps: int = 50000
    batch_size: int = 128


class RecoveryTrainer:
    def __init__(self, config: RecoveryTrainerConfig):
        self.config = config

    def train(self) -> dict[str, float]:
        return {"recovery_steps": float(self.config.total_steps)}
