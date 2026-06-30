# robust-recovery

Research scaffold for robust robotic manipulation recovery under visual, state, and task anomalies.

The repository is organized around four recovery families:

- `baseline_anomalies`: detect anomalies and evaluate nominal policy degradation.
- `actor_critic`: use value/risk estimates to trigger learned recovery.
- `vlm_vla`: combine VLM failure detection with VLA-style corrective actions.
- `hybrid`: fuse rule-based, critic-based, and VLM/VLA recovery signals.

## Quick Start

```bash
cd /home/fmc3-8/workspace/Chen/robust-recovery
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_experiment.py --config configs/train.yaml
```

## Layout

- `configs/`: environment, method, and training YAML files.
- `scripts/`: command-line entry points for data collection, training, and evaluation.
- `src/robust_recovery/`: reusable Python package.
- `experiments/`: per-method experiment outputs and overrides.
- `data/`: nominal, anomaly, recovery, and demonstration datasets.
- `checkpoints/`: trained actors, critics, recovery policies, and anomaly detectors.
- `results/`: logs, figures, videos, and tables.
- `docs/`: method notes and experiment planning.

## Development

The initial code is intentionally lightweight. Replace placeholder model logic with concrete ManiSkill environments, PyTorch networks, VLM clients, and VLA policies as the experiments become defined.

## Current ManiSkill Workflow

See [docs/maniskill_data_collection.md](docs/maniskill_data_collection.md) for the current PickCube data collection commands, validation steps, input/output description, and saved `.npz` format.
