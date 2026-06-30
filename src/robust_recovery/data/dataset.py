"""Dataset helpers for saved trajectories."""

from pathlib import Path


class TrajectoryDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        import numpy as np

        return np.load(self.files[index], allow_pickle=True)
