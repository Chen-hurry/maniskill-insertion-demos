# Scripts README

This directory contains command-line tools for ManiSkill smoke tests, nominal data collection, and collision diagnostics.

## Environment

Run commands from the repository root:

```bash
conda activate maniskill_mp
cd /home/fmc3-8/workspace/Chen/maniskill-insertion-demos
```

The recommended control mode for motion-planning collection is `pd_joint_pos`.

## `collect_nominal_data.py`

Collects nominal demonstrations using `PandaPickPlacePlanner` from `src/maniskill_insertion_demos/planning/panda_motion_planner.py`. The script records observations, planned actions, rewards, done flags, infos, metadata, RGB frames, and optional videos.

### Basic Multi-Camera PickCube

```bash
python scripts/collect_nominal_data.py \
  --env-id PickCubeMultiCam-v1 \
  --robot-uids panda \
  --obs-mode rgb+state \
  --control-mode pd_joint_pos \
  --reward-mode normalized_dense \
  --episodes 1 \
  --max-steps 300 \
  --output-dir data/nominal/pickcube_motionplanning_multicam \
  --save-rgb-frames \
  --save-videos \
  --expected-cameras base_camera top_camera side_camera wrist_camera
```

### Fixed 3x3 Tray Placement

`PickCubeBoxMultiCam-v1` adds a blue 3x3 tray with collision. Each cell has `0.042m x 0.042m` inner size. The red cube is `0.04m`, so the task is intentionally precise.

```bash
python scripts/collect_nominal_data.py \
  --env-id PickCubeBoxMultiCam-v1 \
  --robot-uids panda \
  --obs-mode rgb+state \
  --control-mode pd_joint_pos \
  --reward-mode normalized_dense \
  --episodes 1 \
  --max-steps 420 \
  --output-dir data/nominal/pickcube_box_center \
  --cube-x -0.06 \
  --cube-y 0.00 \
  --cube-yaw 0.0 \
  --goal-x 0.03 \
  --goal-y 0.00 \
  --goal-z 0.024 \
  --save-rgb-frames \
  --save-videos \
  --expected-cameras base_camera top_camera side_camera wrist_camera
```

`goal-z=0.024` corresponds to tray floor thickness `0.004m` plus cube half-size `0.020m`.

### 3x3 Goal Grid Sweep

Use `--goal-grid-3x3` to place the green target at all 9 tray cell centers. If `--episodes 1` is used, the script expands it to 9 episodes automatically.

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

Default grid settings:

```text
goal_grid_center = (0.03, 0.0)
goal_grid_spacing = 0.046
goal_grid_z = 0.024
```

The spacing is `0.042m` cell size plus `0.004m` wall thickness.



### Stable Two-Panda Phone Slot Handoff

`TwoPandaPhoneSlotMultiCam-v1` is the current stable phone insertion task. Use the following command as the baseline before collecting variants:

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

Important notes:

- `--two-panda-mode handoff` is the stable two-arm mode.
- `--handoff-angle-deg 45` is the verified handoff angle. Tests at 50 and 60 degrees were less stable.
- `--handoff-receive-mode upper_side --upper-side-receive-fraction 0.08` is the verified receiving grasp.
- Keep `cube=(-0.08, 0.0)` and `goal=(0.06, 0.0)` for the nominal baseline.
- Videos include a lower-left `S`/`F` marker. `summary.json` and `success_report.json` record the final success rate.

For more visual data diversity, prefer camera/image/light/color perturbations or very small pose offsets. Large scene translations are not recommended for this scripted planner because the robot bases remain fixed and the handoff contact geometry changes.

### Sweeping Initial Conditions

Use sweep options to vary cube position, cube yaw, robot qpos offset, and goal position:

```bash
python scripts/collect_nominal_data.py \
  --env-id PickCubeBoxMultiCam-v1 \
  --robot-uids panda \
  --obs-mode rgb+state \
  --control-mode pd_joint_pos \
  --reward-mode normalized_dense \
  --episodes 1 \
  --max-steps 420 \
  --output-dir data/nominal/pickcube_box_sweep \
  --cube-x-values -0.06 -0.04 \
  --cube-y-values -0.02 0.00 0.02 \
  --cube-yaw-values 0.0 0.4 \
  --goal-grid-3x3 \
  --robot-qpos-offsets \
    "0,0,0,0,0,0,0" \
    "0,0.05,0,0,0,0,0" \
  --save-rgb-frames \
  --save-videos
```

When a sweep option has multiple values and `--episodes 1` is used, the script expands the episode count to cover every combination.

### Return-Home Behavior

By default, collected trajectories end after the robot returns to a Panda home posture:

```text
[0.0, 0.392699, 0.0, -1.963495, 0.0, 2.356194, 0.785398]
```

Disable this with:

```bash
--no-return-home
```

Use a custom 7-DoF arm target with:

```bash
--home-qpos "0,0.392699,0,-1.963495,0,2.356194,0.785398"
```

### Useful Planner Options

- `--planner-place-height`: manually set TCP place height above the goal. The 3x3 tray defaults lower than the normal PickCube scene.
- `--planner-joint-vel-limits`: scale joint velocity limits.
- `--planner-joint-acc-limits`: scale joint acceleration limits.
- `--prefer-rrt`: prefer RRTConnect before screw motion.
- `--enable-grasp-diversity`: allow alternate closing directions and wrist rotations.
- `--disable-grasp-diversity`: force the default stable approach.
- `--refine-scale`: multiply refine/open/close/return-home step counts.

## `test_box_collision_drop.py`

Runs a headless free-drop diagnostic in `PickCubeBoxMultiCam-v1`. The default object is a red dynamic cube with collision. This is useful for checking whether the blue tray walls and floor are active and for studying bad insertion conditions.

### Bad Position And Bad Orientation

Default values intentionally drop the cube near a cell edge with nonzero roll, pitch, and yaw:

```bash
python scripts/test_box_collision_drop.py \
  --output-dir data/diagnostics/bad_cube_drop \
  --steps 180 \
  --save-videos
```

### Center Correct, Yaw Wrong

This tests the case where the target center is correct but the cube orientation is slightly wrong:

```bash
python scripts/test_box_collision_drop.py \
  --output-dir data/diagnostics/center_correct_yaw_error \
  --steps 180 \
  --drop-x 0.03 \
  --drop-y 0.0 \
  --target-x 0.03 \
  --target-y 0.0 \
  --drop-roll 0.0 \
  --drop-pitch 0.0 \
  --drop-yaw 0.10 \
  --save-videos
```

For the tight 4.2cm cells, even a small yaw error can cause the 4cm cube to hit the tray wall or divider and stop above the floor.

### Perfect Center Baseline

```bash
python scripts/test_box_collision_drop.py \
  --output-dir data/diagnostics/center_correct_yaw_zero \
  --steps 180 \
  --drop-x 0.03 \
  --drop-y 0.0 \
  --target-x 0.03 \
  --target-y 0.0 \
  --drop-roll 0.0 \
  --drop-pitch 0.0 \
  --drop-yaw 0.0 \
  --save-videos
```

Even this free-fall baseline can fail to settle on the tray floor because impact can tilt the cube. This is expected for a very tight tray; stable insertion should be controlled and slow.

### Optional Sphere Test

```bash
python scripts/test_box_collision_drop.py \
  --drop-shape sphere \
  --sphere-radius 0.02 \
  --drop-x 0.03 \
  --drop-y 0.0 \
  --save-videos
```

## `test_maniskill_env.py`

Smoke-tests a ManiSkill environment with random actions. Use it only to validate environment creation, observation format, camera output, and video saving.

```bash
python scripts/test_maniskill_env.py \
  --env-id PickCubeMultiCam-v1 \
  --robot-uids panda \
  --obs-mode rgb+state \
  --control-mode pd_joint_pos \
  --reward-mode normalized_dense \
  --episodes 1 \
  --max-steps 100 \
  --output-dir results/pickcube_multicam_smoke \
  --save-rgb-frames \
  --save-videos \
  --expected-cameras base_camera top_camera side_camera wrist_camera
```

## Outputs

Most scripts write:

```text
output_dir/
├── episodes/episode_000000.npz
├── summary.json
├── rgb_frames/<camera>/episode_000000/frame_000000.png
└── videos/<camera>/episode_000000.mp4
```

Diagnostics write a similar `summary.json` plus optional camera videos.
