"""Small logging helper."""

import logging


def get_logger(name: str = "robust_recovery") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)
