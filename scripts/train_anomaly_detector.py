#!/usr/bin/env python
"""Train a visual or multimodal anomaly detector."""

import argparse

from robust_recovery.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/method/anomalies.yaml")
    parser.add_argument("--data-dir", default="data/nominal")
    args = parser.parse_args()

    get_logger(__name__).info("Training anomaly detector with %s on %s", args.config, args.data_dir)


if __name__ == "__main__":
    main()
