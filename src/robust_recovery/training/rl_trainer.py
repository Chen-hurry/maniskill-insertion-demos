"""Generic RL trainer scaffold."""

from dataclasses import dataclass


@dataclass
class RLTrainerConfig:
    total_steps: int = 100000
    batch_size: int = 128
    learning_rate: float = 3e-4


class RLTrainer:
    def __init__(self, config: RLTrainerConfig):
        self.config = config

    def train(self) -> dict[str, float]:
        return {"steps": float(self.config.total_steps)}
