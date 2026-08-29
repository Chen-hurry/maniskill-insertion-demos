# Data Directory

This directory stores generated trajectories, RGB videos, diagnostics, metadata, and dataset-viewer artifacts for robust manipulation and phone-slot handoff experiments.

Large generated datasets should stay out of git unless a small fixture is intentionally needed. Keep this README and lightweight configuration in version control.

## Layout

```text
data/
├── nominal/       # Successful or intended nominal demonstrations
├── diagnostics/   # Collision, camera, and environment sanity checks
├── anomalies/     # Future anomaly rollouts or injected-failure data
├── recovery/      # Future recovery rollouts after anomaly states
└── demos/         # Future expert, teleop, or corrected demonstrations
```

## Episode Format

`scripts/collect_nominal_data.py` writes one `.npz` file per episode plus summary files:

```text
data/nominal/<dataset_name>/
├── episodes/
│   ├── episode_000000.npz
│   └── episode_000001.npz
├── summary.json
├── success_report.json
├── trajectory_diversity.json        # optional, generated separately
├── combined/
│   └── combined_all_cameras.mp4
├── videos/
│   ├── base_camera/
│   ├── top_camera/
│   ├── side_camera/
│   ├── left_wrist_camera/
│   └── right_wrist_camera/
├── viewer_episode_000000.html       # optional HTML viewer
└── demo_videos/
    └── episode_000000_pi05_demo.mp4 # optional annotated demo video
```

Each episode `.npz` contains:

```text
episode_id
num_steps
observations
actions
rewards
dones
infos
metadata
```

Important fields:

- `observations`: RGB observations and flattened state.
- `actions`: raw environment actions used by the current controller.
- `metadata.language_instruction`: task instruction for VLA or PI0.5 style training.
- `metadata.initial_state_config`: slot id, phone pose, support-platform pose, robot qpos offset, and planner options.
- `metadata.planner_result`: planner stages and diagnostic information.
- `metadata.env_success`: final environment success flag.

For PI0.5 training, the raw `actions` are still environment `pd_joint_pos` actions. If training a 14D absolute end-effector action policy, export labels as:

```text
left_tcp_xyz + left_tcp_rpy + left_gripper
right_tcp_xyz + right_tcp_rpy + right_gripper
```

That is `6 + 1 + 6 + 1 = 14` dimensions. The current dataset stores enough state to derive TCP poses, but a dedicated converter should be used before PI0.5 training.

## Current Phone-Slot Dataset

The current demonstration dataset for PI0.5 inspection is:

```text
data/nominal/pi05_phone_27_success_video/
```

It is generated from `TwoPandaPhoneSlotMultiCam-v1` with:

- 3 target slots: code `slot_id = 0, 1, 2`, displayed as slots `1, 2, 3`.
- 3 phone/support-platform positions: `cube_y = -0.02, 0.0, 0.02`.
- 3 robot initial qpos offsets.
- Total combinations: `3 x 3 x 3 = 27`.
- All saved episodes are successful.
- Videos are saved for `base_camera`, `top_camera`, `side_camera`, `left_wrist_camera`, and `right_wrist_camera`.

The stable handoff logic currently includes:

- Slot-aware insert arm selection.
- A fixed `handoff_center` stage before flipping.
- Phone/support-platform movement through `--move-conveyor-with-cube`.
- Left receive confirmation before releasing the right hand.
- Left receive candidate priority at `planner_left_receive_primary_fraction = 0.45`.

## Collect 27 Successful PI0.5 Phone Episodes

Run from the repository root:

```bash
cd /path/to/maniskill-insertion-demos

conda run -n maniskill_mp python scripts/collect_nominal_data.py \
  --env-id TwoPandaPhoneSlotMultiCam-v1 \
  --robot-uids panda panda \
  --obs-mode rgb+state \
  --control-mode pd_joint_pos \
  --reward-mode normalized_dense \
  --episodes 27 \
  --target-successes 27 \
  --max-attempts 81 \
  --save-only-successful \
  --save-videos \
  --video-fps 20 \
  --max-steps 1800 \
  --output-dir data/nominal/pi05_phone_27_success_video \
  --two-panda-mode handoff \
  --handoff-angle-deg 45 \
  --handoff-receive-mode upper_side \
  --upper-side-receive-fraction 0.08 \
  --slot-ids 0 1 2 \
  --cube-xy-values=-0.08,-0.02 \
  --cube-xy-values=-0.08,0.0 \
  --cube-xy-values=-0.08,0.02 \
  --move-conveyor-with-cube \
  --robot-qpos-offsets=0,0,0,0,0,0,0 \
  --robot-qpos-offsets=0,0.02,0,0,0,0,0 \
  --robot-qpos-offsets=0,-0.02,0,0,0,0,0 \
  --language-instruction-template "Pick up the phone, hand it over, and insert it into slot {slot_number}." \
  --print-planner-stages \
  --expected-cameras base_camera top_camera side_camera left_wrist_camera right_wrist_camera
```

The `--target-successes 27 --save-only-successful` pair keeps attempting until 27 successful episodes have been saved, up to `--max-attempts`.

## Evaluate Trajectory Similarity

After collection, compute pairwise trajectory diversity and similarity:

```bash
conda run -n maniskill_mp python scripts/evaluate_trajectory_diversity.py \
  --dataset-dir data/nominal/pi05_phone_27_success_video \
  --num-samples 200 \
  --gripper-weight 0.25 \
  --similarity-tau 0.01 \
  --group-key initial_state_config.slot_id \
  --output data/nominal/pi05_phone_27_success_video/trajectory_diversity.json
```

Useful output fields:

- `trajectory_similarity.pairwise_summary_excluding_self.joint_similarity_mean`
- `trajectory_similarity.pairwise_summary_excluding_self.combined_similarity_mean`
- `grouped_similarity.weighted_summary.joint_diversity_mean`
- `grouped_similarity.groups`

## Dataset Viewers

### Interactive HTML Viewer

Build a browser-based viewer for a single episode:

```bash
conda run -n maniskill_mp python scripts/build_pi05_dataset_viewer.py \
  --dataset-dir data/nominal/pi05_phone_27_success_video \
  --episode-id 0
```

Output:

```text
data/nominal/pi05_phone_27_success_video/viewer_episode_000000.html
```

Open it locally on the server with:

```bash
xdg-open /path/to/maniskill-insertion-demos/data/nominal/pi05_phone_27_success_video/viewer_episode_000000.html
```

The HTML viewer shows four synchronized videos:

```text
top_camera
left_wrist_camera
right_wrist_camera
side_camera
```

Below the videos it displays the current frame's phone pose, left/right TCP pose, goal vector, action summary, and language instruction.

### Annotated Demo Video

Build a single MP4 with four videos on top and a white information panel below:

```bash
conda run -n maniskill_mp python scripts/build_pi05_dataset_demo_video.py \
  --dataset-dir data/nominal/pi05_phone_27_success_video \
  --episode-id 0
```

Output:

```text
data/nominal/pi05_phone_27_success_video/demo_videos/episode_000000_pi05_demo.mp4
```

The lower white panel includes:

- Language instruction.
- Frame index, slot id, reward, and done flag.
- Phone pose.
- Left and right TCP pose.
- TCP-to-phone vectors.
- Goal position and phone-to-goal vector.
- Raw action summary.

Generate all 27 annotated demo videos:

```bash
for i in $(seq -w 0 26); do
  conda run -n maniskill_mp python scripts/build_pi05_dataset_demo_video.py \
    --dataset-dir data/nominal/pi05_phone_27_success_video \
    --episode-id "$i"
done
```

## Copy Videos To A Local Machine

From the local computer, download one demo video with `scp`:

```bash
scp <user>@<host>:/path/to/maniskill-insertion-demos/data/nominal/pi05_phone_27_success_video/demo_videos/episode_000000_pi05_demo.mp4 .
```

The final `.` means "save to the current local directory."

## Diagnostics

`scripts/test_box_collision_drop.py` writes diagnostic summaries and optional videos:

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

- Use `summary.json` first when checking success rates.
- Use `success_report.json` for a compact success/failure report.
- Use `combined/combined_all_cameras.mp4` for a fast visual sanity check.
- Use `trajectory_diversity.json` for pairwise trajectory similarity and diversity.
- Use `demo_videos/*_pi05_demo.mp4` for presentations and dataset explanation.
- Keep large generated datasets and videos out of git unless explicitly requested.
