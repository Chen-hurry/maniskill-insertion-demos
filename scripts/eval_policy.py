#!/usr/bin/env python
"""Evaluate nominal and recovery policies."""

import argparse

from robust_recovery.evaluation.evaluator import Evaluator
from robust_recovery.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    metrics = Evaluator(episodes=args.episodes).evaluate()
    get_logger(__name__).info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
