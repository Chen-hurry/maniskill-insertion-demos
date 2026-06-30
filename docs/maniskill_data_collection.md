# ManiSkill Data Collection Guide

This document records the current first-stage workflow:

1. Install and activate the ManiSkill environment.
2. Run `PickCube-v1` with a random policy.
3. Save nominal trajectories.
4. Verify saved data.
5. Understand the input, output, and file format.

## 1. Environment Setup

Use the `maniskill` conda environment:

```bash
conda activate maniskill
```

Install this project in editable mode from the repository root:

```bash
cd /home/fmc3-8/workspace/Chen/robust-recovery
pip install -e .
```

Editable install means Python imports `robust_recovery` directly from this source tree. If files under `src/robust_recovery/` change, no reinstall is needed.

For development and tests, install the optional dev dependencies:

```bash
pip install -e ".[dev]"
```

## 2. Basic ManiSkill Verification

Verify that ManiSkill can start a simple task:

```bash
python -m mani_skill.examples.demo_random_action -e PickCube-v1
```

A successful run prints the observation space, action space, reward, termination flags, and task info for each step. Random actions usually do not solve the task, so `success: False` is normal.

## 3. Nominal Data Collection

The first collection script is:

```bash
scripts/test_maniskill_env.py
```

Run a small smoke test:

```bash
python scripts/test_maniskill_env.py \
  --env-id PickCube-v1 \
  --obs-mode state \
  --episodes 1 \
  --max-steps 5 \
  --output-dir data/nominal/pick_cube_smoke
```

Collect 10 nominal trajectories:

```bash
python scripts/test_maniskill_env.py \
  --env-id PickCube-v1 \
  --obs-mode state \
  --episodes 10 \
  --max-steps 50 \
  --output-dir data/nominal/pick_cube
```

Recommended first-stage settings:

- `--env-id PickCube-v1`
- `--obs-mode state`
- `--episodes 10`
- `--max-steps 50`
- `--output-dir data/nominal/pick_cube`

Use `state` first because the current machine has not yet been confirmed stable for RGBD/Vulkan rendering.

## 4. RGBD and Video Collection

After Vulkan/GPU rendering is working, try RGBD:

```bash
python scripts/test_maniskill_env.py \
  --env-id PickCube-v1 \
  --obs-mode rgbd \
  --episodes 10 \
  --max-steps 50 \
  --output-dir data/nominal/pick_cube_rgbd \
  --save-videos
```

Video files are written under:

```text
data/nominal/pick_cube_rgbd/videos/
```

If this command reports Vulkan, NVIDIA, or rendering errors, keep using `--obs-mode state` and fix rendering separately.

## 5. Script Inputs

`scripts/test_maniskill_env.py` accepts:

```text
--env-id          ManiSkill task id. Default: PickCube-v1
--obs-mode        Observation mode. Default: state
--control-mode    Control mode. Default: pd_ee_delta_pose
--reward-mode     Reward mode. Default: normalized_dense
--render-mode     Optional render mode. Default: None
--episodes        Number of episodes to collect. Default: 10
--max-steps       Max steps per episode. Default: 50
--output-dir      Directory for saved trajectory files.
--save-videos     Save videos when RGB frames exist in observations.
--video-fps       Video frame rate. Default: 20
```

The policy is currently random:

```python
action = env.action_space.sample()
```

This is intentional for the first stage. The goal is to verify the environment, data pipeline, and storage format before using a learned policy.

## 6. Saved Output

Each episode is saved as one compressed NumPy file:

```text
data/nominal/pick_cube/episode_000000.npz
data/nominal/pick_cube/episode_000001.npz
...
```

The filename format is:

```text
episode_<six-digit-episode-id>.npz
```

For example:

```text
episode_000007.npz
```

## 7. `.npz` File Format

Each `.npz` contains:

```text
episode_id      scalar episode id
num_steps       number of executed actions
observations    object array of observations
actions         object array of actions
rewards         float32 array, shape [num_steps]
dones           bool array, shape [num_steps]
infos           object array of ManiSkill info dictionaries
```

Important length convention:

- `actions`, `rewards`, and `dones` have length `num_steps`.
- `observations` has length `num_steps + 1`, because it includes the reset observation before the first action.
- `infos` includes the reset info plus step infos.

For `--obs-mode state`, observations are low-dimensional state arrays.

For `--obs-mode rgbd`, observations may contain nested dictionaries with camera images, depth, segmentation, and state fields depending on the ManiSkill task.

## 8. Inspect Saved Data

Print the saved keys and shapes:

```bash
python - <<'PY'
import numpy as np

path = "data/nominal/pick_cube/episode_000000.npz"
data = np.load(path, allow_pickle=True)

print("keys:", data.files)
print("episode_id:", data["episode_id"].item())
print("num_steps:", data["num_steps"].item())
print("observations:", len(data["observations"]))
print("actions:", len(data["actions"]))
print("rewards shape:", data["rewards"].shape)
print("dones shape:", data["dones"].shape)
PY
```

Expected relationship:

```text
len(observations) = len(actions) + 1
len(rewards) = len(actions)
len(dones) = len(actions)
```

## 9. Test and Validation Commands

Run helper tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_maniskill_env_helpers.py -q
```

If running through conda without activating the environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n maniskill pytest tests/test_maniskill_env_helpers.py -q
```

Run Python compile checks:

```bash
python -m compileall -q scripts src tests
```

Or through conda:

```bash
conda run -n maniskill python -m compileall -q scripts src tests
```

## 10. Current First-Stage Goal

The current milestone is:

```text
Run PickCube-v1 in ManiSkill and save 10 nominal trajectories.
```

The expected output is:

```text
data/nominal/pick_cube/
├── episode_000000.npz
├── episode_000001.npz
├── ...
└── episode_000009.npz
```

After this is stable, the next steps are:

1. Add simple anomaly injection, such as object displacement.
2. Save anomaly trajectories under `data/anomalies/pick_cube/`.
3. Build a simple nearest-neighbor visual or state anomaly detector.
4. Add rule-based recovery triggers.
