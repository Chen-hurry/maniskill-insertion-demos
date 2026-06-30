#!/usr/bin/env python
"""Run a configured robust recovery experiment."""

import argparse
from pathlib import Path

import yaml

from robust_recovery.utils.logger import get_logger
from robust_recovery.utils.seed import set_seed


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))
    logger = get_logger(__name__)
    logger.info("Loaded experiment config from %s", args.config)
    logger.info("Output directory: %s", config.get("output_dir", "experiments/default"))


if __name__ == "__main__":
    main()
