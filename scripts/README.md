## `scripts/test_maniskill_env.py`

### English

This script is a smoke test for ManiSkill environments. It launches a specified ManiSkill task, executes a random policy sampled from the environment action space, and saves the collected trajectory data.

For each episode, the script records:

- observations
- actions
- rewards
- done flags
- environment infos
- RGB frames, if visual observations are available
- MP4 videos, if `--save-videos` is enabled

The script is mainly used to verify that the ManiSkill environment, observation mode, control mode, data saving pipeline, and video recording pipeline work correctly before training or collecting large-scale datasets.

Example:

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
  --save-videos
  --expected-cameras base_camera top_camera side_camera wrist_camera
```

## `scripts/collect_nominal_data.py`

### English

This script collects nominal successful trajectories from ManiSkill environments using a motion-planning-based Panda policy. Unlike `scripts/test_maniskill_env.py`, which uses a random policy, this script calls the project motion-planning wrapper in `src/robust_recovery/planning/panda_motion_planner.py` to generate planned pick-and-place motions.

For each episode, the script records:

- observations
- planned joint-position actions
- rewards
- done flags
- environment infos
- RGB frames from all available cameras
- MP4 videos, if `--save-videos` is enabled
- planning metadata, including initial object position, goal position, and planner result

The script is mainly used to collect nominal data for robust recovery experiments. The collected trajectories can be used as normal successful demonstrations, replay references, or baseline data for later anomaly detection and recovery policy training.

Because the ManiSkill motion planner executes joint-space trajectories, the recommended control mode is `pd_joint_pos`.

Example:

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

### Enumerating initial states

Use these options when you want the dataset diversity to come from different
initial robot states and red-cube positions instead of wrist rotation during
the approach:

```bash
python scripts/collect_nominal_data.py \
  --env-id PickCubeMultiCam-v1 \
  --robot-uids panda \
  --obs-mode rgb+state \
  --control-mode pd_joint_pos \
  --reward-mode normalized_dense \
  --episodes 1 \
  --max-steps 300 \
  --output-dir data/nominal/pickcube_motionplanning_multicam_sweep \
  --cube-x-values -0.03 0.00 0.03 \
  --cube-y-values -0.03 0.00 0.03 \
  --robot-qpos-offsets \
    "0,0,0,0,0,0,0" \
    "0,0.05,0,0,0,0,0" \
    "0,-0.05,0,0,0,0,0" \
  --save-rgb-frames \
  --save-videos \
  --expected-cameras base_camera top_camera side_camera wrist_camera
```

When any sweep option has multiple values and `--episodes 1` is used, the
script automatically expands the episode count to cover every combination. The
example above collects `3 x 3 x 3 = 27` episodes.

Each episode stores the applied `robot_qpos_offset`, actual `robot_qpos`, and
`cube_pose` in `summary.json` and in the episode metadata.
