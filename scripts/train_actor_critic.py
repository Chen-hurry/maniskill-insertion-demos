#!/usr/bin/env python
"""Train nominal actor-critic policy components."""

import argparse

from robust_recovery.training.rl_trainer import RLTrainer, RLTrainerConfig
from robust_recovery.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()

    metrics = RLTrainer(RLTrainerConfig(total_steps=args.steps)).train()
    get_logger(__name__).info("Actor-critic training finished: %s", metrics)


if __name__ == "__main__":
    main()
