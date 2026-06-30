"""Visualization helpers for result figures and videos."""

from pathlib import Path


def ensure_figure_dir(path: str | Path) -> Path:
    figure_dir = Path(path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir
