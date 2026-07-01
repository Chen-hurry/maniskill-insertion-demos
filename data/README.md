# Data Directory

This directory stores trajectories, RGB frames, videos, diagnostics, and metadata for robust manipulation recovery experiments.

## Layout

```text
data/
├── nominal/       # Successful or intended nominal demonstrations
├── diagnostics/   # Collision, camera, and environment sanity checks
├── anomalies/     # Future anomaly rollouts or injected-failure data
├── recovery/      # Future recovery rollouts after anomaly states
└── demos/         # Future expert, teleop, or corrected demonstrations
```

Generated data directories are usually ignored by git. Keep only lightweight documentation and configuration in version control unless a small fixture is intentionally needed.

## Nominal Motion-Planning Data

`collect_nominal_data.py` writes one `.npz` per episode and a `summary.json` file:

```text
data/nominal/<dataset_name>/
├── episodes/
│   ├── episode_000000.npz
│   └── episode_000001.npz
├── summary.json
├── rgb_frames/
│   ├── base_camera/
│   ├── top_camera/
│   ├── side_camera/
│   └── wrist_camera/
└── videos/
    ├── base_camera/
    ├── top_camera/
    ├── side_camera/
    └── wrist_camera/
```

Each episode file contains:

- `episode_id`
- `num_steps`
- `observations`
- `actions`
- `rewards`
- `dones`
- `infos`
- `metadata`

The metadata records the environment id, camera set, seed, initial cube pose, goal pose, applied robot qpos offset, planner settings, planner result, and whether the environment reported success.

## 3x3 Tray Datasets

Recommended directory names:

```text
data/nominal/pickcube_box_center/
data/nominal/pickcube_box_grid_3x3/
data/nominal/pickcube_box_sweep/
```

Use `PickCubeBoxMultiCam-v1` when collecting data for precise placement into the blue 3x3 tray. The tray cell inner size is `0.042m x 0.042m`; the cube size is `0.04m`, so failures from small pose errors are expected and useful.



## Phone-Slot Handoff Datasets

Recommended stable phone-slot dataset name:

```text
data/nominal/two_panda_phone_slot_handoff_stable/
```

This dataset should be collected with `TwoPandaPhoneSlotMultiCam-v1` and the stable 45 degree handoff configuration. Each run writes:

```text
summary.json
success_report.json
videos/<camera>/episode_000000.mp4
combined/combined_all_cameras.mp4
episodes/episode_000000.npz
```

For quick visual checks, open `combined/combined_all_cameras.mp4`. The lower-left marker shows `S` for success and `F` for failure. Keep large generated datasets out of git.

## Diagnostics

`test_box_collision_drop.py` writes diagnostic summaries and optional videos:

```text
data/diagnostics/<diagnostic_name>/
├── summary.json
└── videos/
    ├── base_camera.mp4
    ├── top_camera.mp4
    ├── side_camera.mp4
    └── wrist_camera.mp4
```

The summary contains object type, initial drop pose, target center, final pose, whether the object remained inside the target cell, and whether it settled near the expected floor height.

## Notes

- Large generated files should stay out of git unless explicitly needed.
- Use `summary.json` first when checking success rates.
- Use `videos/top_camera.mp4` first when diagnosing tray insertion and collision behavior.
