# maniskill-insertion-demos

This repository is a working scaffold for robust robotic manipulation recovery experiments on ManiSkill. The current implementation focuses on Panda `PickCube` data collection with multi-camera observations, scripted motion-planning demonstrations, and a fine placement scene where a red cube is inserted into a small 3x3 tray.

本仓库用于鲁棒机器人操作恢复实验。目前主要围绕 ManiSkill 中的 Panda `PickCube` 任务，支持多相机采集、运动规划示范数据、固定目标点、3x3 精细盒子场景，以及碰撞/放置诊断脚本。

## Current Workflow

The active workflow is:

1. Use custom ManiSkill environments in `src/maniskill_insertion_demos/envs/`.
2. Collect nominal pick-and-place demonstrations with `scripts/collect_nominal_data.py`.
3. Save `rgb+state` observations, planned actions, rewards, infos, metadata, RGB frames, and videos.
4. Sweep cube positions, goal positions, robot initial offsets, or a built-in 3x3 goal grid.
5. Use diagnostic drop tests to inspect whether the 4cm cube can enter the 4.2cm tray cells.

## Key Features

- `PickCubeMultiCam-v1`: PickCube with base, top, side, and wrist cameras.
- `PickCubeBoxMultiCam-v1`: PickCube with a visible blue 3x3 collision tray on the table.
- Visible green goal marker for target visualization.
- Red cube size: `0.04m x 0.04m x 0.04m`.
- Tray cell inner size: `0.042m x 0.042m`, leaving about `0.001m` clearance per side.
- Motion-planning collection with Panda pick, place, open gripper, and optional return-home stage.
- Headless RGB frame and video saving for remote-server workflows.
- Diagnostic free-drop script for testing collision behavior with a red cube or sphere.

## Quick Start

Use the existing ManiSkill conda environment:

```bash
conda activate maniskill_mp
cd /path/to/maniskill-insertion-demos
```

Collect one 3x3 tray sweep. With `--goal-grid-3x3`, `--episodes 1` automatically expands to 9 episodes:

```bash
python scripts/collect_nominal_data.py \
  --env-id PickCubeBoxMultiCam-v1 \
  --robot-uids panda \
  --obs-mode rgb+state \
  --control-mode pd_joint_pos \
  --reward-mode normalized_dense \
  --episodes 1 \
  --max-steps 420 \
  --output-dir data/nominal/pickcube_box_grid_3x3 \
  --cube-x -0.06 \
  --cube-y 0.00 \
  --cube-yaw 0.0 \
  --goal-grid-3x3 \
  --save-rgb-frames \
  --save-videos \
  --expected-cameras base_camera top_camera side_camera wrist_camera
```

Run the collision diagnostic with a red cube whose center is correct but yaw is slightly wrong:

```bash
python scripts/test_box_collision_drop.py \
  --output-dir data/diagnostics/center_correct_yaw_error \
  --drop-x 0.03 \
  --drop-y 0.0 \
  --target-x 0.03 \
  --target-y 0.0 \
  --drop-roll 0.0 \
  --drop-pitch 0.0 \
  --drop-yaw 0.10 \
  --save-videos
```



## Stable Phone-Slot Handoff

The current stable phone-slot collection path uses `TwoPandaPhoneSlotMultiCam-v1`. The right Panda grasps the phone, flips it to a 45 degree handoff pose, the left Panda receives it with the `upper_side` grasp, calibrates to upright, inserts it into the slot, releases, and both arms return home.

当前稳定的手机插槽任务使用 `TwoPandaPhoneSlotMultiCam-v1`。不要大幅移动手机和槽位来制造图像多样性；这会改变机器人相对物体的几何关系，导致规则 planner 夹取/交接失败。稳定基准命令如下：

```bash
conda run -n maniskill_mp python scripts/collect_nominal_data.py \
  --env-id TwoPandaPhoneSlotMultiCam-v1 \
  --robot-uids panda panda \
  --obs-mode rgb+state \
  --control-mode pd_joint_pos \
  --reward-mode normalized_dense \
  --episodes 1 \
  --max-steps 1800 \
  --output-dir data/nominal/two_panda_phone_slot_handoff_stable \
  --two-panda-mode handoff \
  --handoff-angle-deg 45 \
  --handoff-receive-mode upper_side \
  --upper-side-receive-fraction 0.08 \
  --cube-x -0.08 \
  --cube-y 0.0 \
  --goal-x 0.06 \
  --goal-y 0.0 \
  --print-planner-stages \
  --save-videos \
  --expected-cameras base_camera top_camera side_camera left_wrist_camera right_wrist_camera
```

Outputs include `summary.json`, `success_report.json`, per-camera videos, and `combined/combined_all_cameras.mp4`. The saved videos mark final status in the lower-left corner: green `S` for success and red `F` for failure.

Recommended diversity sources for rule-based collection:

- Keep the physical task near the stable pose and collect only successful rollouts.
- Add small pose perturbations around the stable point, e.g. millimeter-level `cube-y` offsets.
- Prefer camera/view, lighting, color, texture, crop, and image augmentation for visual diversity.
- Avoid large phone/slot translations unless the robot base and planner targets are redesigned together.

## Repository Layout

```text
maniskill-insertion-demos/
├── data/                        # Collected datasets and diagnostics
├── docs/                        # Data-collection notes
├── scripts/                     # Collection, filtering, and visualization entry points
├── src/maniskill_insertion_demos/         # Python package code
│   ├── envs/                    # Custom ManiSkill environments
│   └── planning/                # Panda motion-planning wrapper
├── tests/                       # Helper unit tests
└── results/                     # Smoke-test outputs
```

## Important Files

- `scripts/collect_nominal_data.py`: motion-planning data collector.
- `scripts/test_box_collision_drop.py`: collision/free-drop diagnostic script.
- `src/maniskill_insertion_demos/envs/pickcube_multicam.py`: custom multi-camera and 3x3 tray environments.
- `src/maniskill_insertion_demos/planning/panda_motion_planner.py`: high-level Panda pick-and-place planner.
- `scripts/filter_idle_dataset.py`: drop idle/near-static frames from a collected dataset.
- `scripts/evaluate_trajectory_diversity.py`: report trajectory diversity across episodes.
- `scripts/build_pi05_dataset_demo_video.py`: multi-camera demo video with PI0.5 field overlay.
- `scripts/build_pi05_dataset_viewer.py`: dataset inspection viewer.
- `docs/maniskill_data_collection.md`: longer notes for the ManiSkill data-collection workflow.

## Notes

The 3x3 tray is intentionally tight. A 4cm cube inside a 4.2cm cell has only about 1mm side clearance, so free-fall tests often fail even when the target center is correct. Stable insertion should use slow, vertical, orientation-aligned placement rather than dropping the cube.
