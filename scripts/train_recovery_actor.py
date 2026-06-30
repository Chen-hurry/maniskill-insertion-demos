#!/usr/bin/env python
"""Train a recovery actor from anomalous and recovery trajectories."""

import argparse

from robust_recovery.training.recovery_trainer import RecoveryTrainer, RecoveryTrainerConfig
from robust_recovery.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()

    metrics = RecoveryTrainer(RecoveryTrainerConfig(total_steps=args.steps)).train()
    get_logger(__name__).info("Recovery actor training finished: %s", metrics)


if __name__ == "__main__":
    main()
