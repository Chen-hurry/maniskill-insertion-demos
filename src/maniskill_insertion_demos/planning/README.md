# Planning Module

This module contains motion-planning wrappers used to generate Panda manipulation trajectories for robust recovery experiments.

## Main File

`panda_motion_planner.py` wraps ManiSkill's official Panda motion-planning solver and adds task-level pick-and-place behavior for this project.

The wrapper uses ManiSkill/MPLib planning for TCP pose moves, then executes actions through the environment so the data collector can record observations, actions, rewards, infos, frames, and videos.

## Pick-And-Place Stages

A nominal trajectory is decomposed into:

```text
current TCP
-> pre_grasp
-> grasp
-> close_gripper
-> lift
-> pre_place
-> place
-> open_gripper
-> return_home       # enabled by default in collect_nominal_data.py
```

The planner result records `completed_stages`, `failed_stage`, selected grasp candidate information, and return-home metadata.

## No-Rotation Approach

By default, the planner keeps the current TCP orientation when approaching the cube. This avoids unnecessary wrist rotation near the red cube and produces smoother, less distracting nominal data.

When grasp diversity is enabled, the planner can test multiple closing directions. This can increase data diversity but may also introduce extra wrist motion.

## Placement Height

For normal PickCube placement, a higher place TCP target is acceptable. For the 3x3 tray scene, the planner automatically uses a lower place height so the cube is inserted closer to the tray floor before opening the gripper.

The collector exposes this as:

```bash
--planner-place-height <float>
```

Leave it unset for the default behavior. The tray goal marker should usually use:

```text
goal-z = 0.024
```

This is `0.004m` tray floor thickness plus `0.020m` cube half-size.

## Return Home

`collect_nominal_data.py` enables return-home by default. After opening the gripper, the planner smoothly interpolates the first 7 Panda arm joints back to:

```text
[0.0, 0.392699, 0.0, -1.963495, 0.0, 2.356194, 0.785398]
```

This makes the end of each collected episode cleaner: the object is placed, the gripper is open, and the arm has moved away from the tray.

Disable return-home with:

```bash
--no-return-home
```

Set a custom 7-DoF target with:

```bash
--home-qpos "0,0.392699,0,-1.963495,0,2.356194,0.785398"
```



## Two-Panda Phone-Slot Planner

`TwoPandaPhoneSlotPlanner` supports the stable handoff path for `TwoPandaPhoneSlotMultiCam-v1`:

```text
right_pre_grasp
-> right_grasp
-> right_close_gripper
-> right_lift_flat
-> right_flip_handoff_45
-> left_pre_receive
-> left_receive
-> left_close_gripper
-> right_release_handoff
-> right_retract_after_handoff
-> left_lift_after_handoff
-> left_calibrate_68
-> left_calibrate_90
-> left_pre_insert
-> left_insert
-> left_open_gripper
-> right_return_home
-> left_return_home
```

The verified configuration is:

```text
two_panda_mode = handoff
handoff_angle_deg = 45
handoff_receive_mode = upper_side
upper_side_receive_fraction = 0.08
phone initial position = (-0.08, 0.0)
slot goal position = (0.06, 0.0)
```

This planner is rule-based and contact-sensitive. It is reliable for the tuned baseline but should not be treated as a globally robust planner over arbitrary phone/slot positions. For broad visual diversity, change camera/rendering conditions or apply image augmentation rather than moving the task far from the tuned geometry.

## Planning Options Exposed By The Collector

- `--planner-joint-vel-limits`: scales joint velocity limits.
- `--planner-joint-acc-limits`: scales joint acceleration limits.
- `--prefer-rrt`: tries RRTConnect before screw planning.
- `--refine-scale`: multiplies refine/open/close/return-home step counts.
- `--planner-place-height`: overrides the default place height.
- `--enable-grasp-diversity`: allows multiple closing directions and rotation candidates.
- `--disable-grasp-diversity`: keeps the default stable grasp behavior.

## Important Behavior For The 3x3 Tray

The tray task is intentionally tight:

```text
cube side length = 0.040m
tray cell inner side length = 0.042m
clearance per side = about 0.001m
```

This means center position alone is not enough. The cube must also be orientation-aligned and lowered slowly. Free-fall diagnostics show that a cube can fail to settle on the floor even when the XY center is correct, because impact can tilt the cube and make it catch on a divider or wall.

Use the collision diagnostic script for quick checks:

```bash
python scripts/test_box_collision_drop.py \
  --drop-x 0.03 \
  --drop-y 0.0 \
  --target-x 0.03 \
  --target-y 0.0 \
  --drop-yaw 0.10 \
  --save-videos
```
