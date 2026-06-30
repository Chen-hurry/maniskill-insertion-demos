#!/usr/bin/env python
"""Collect nominal PickCube trajectories with ManiSkill motion planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sapien
from tqdm import trange

from robust_recovery.planning.panda_motion_planner import PandaPickPlacePlanner


def to_numpy_tree(value: Any) -> Any:
    """Convert tensors and nested containers into numpy-friendly objects."""
    if value is None or isinstance(value, (str, bytes, int, float, bool, Path)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        return {key: to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [to_numpy_tree(item) for item in value]
        try:
            array = np.asarray(converted)
        except ValueError:
            return converted
        if array.dtype == object:
            return converted
        return array
    return np.asarray(value)


def _as_bool(value: Any) -> bool:
    value = to_numpy_tree(value)
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


def scalar_float(value: Any) -> float:
    value = to_numpy_tree(value)
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def parse_float_csv(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_robot_qpos_offsets(values: list[str] | None) -> list[np.ndarray]:
    if not values:
        return [np.zeros(7, dtype=np.float32)]

    offsets: list[np.ndarray] = []
    for value in values:
        offset = np.asarray(parse_float_csv(value), dtype=np.float32)
        if offset.size not in (7, 9):
            raise ValueError(
                "Each --robot-qpos-offsets value must contain 7 arm offsets "
                f"or 9 full qpos offsets, got {offset.size}: {value}"
            )
        offsets.append(offset)
    return offsets


def parse_optional_home_qpos(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    qpos = np.asarray(parse_float_csv(value), dtype=np.float32)
    if qpos.size != 7:
        raise ValueError(f"--home-qpos must contain 7 arm joint values, got {qpos.size}: {value}")
    return qpos


def single_or_sweep_values(single_value: float | None, sweep_values: list[float] | None) -> list[float | None]:
    if sweep_values is not None:
        return sweep_values
    if single_value is not None:
        return [single_value]
    return [None]


def goal_position_values(args: argparse.Namespace) -> tuple[list[float | None], list[float | None], list[float | None]]:
    if not args.goal_grid_3x3:
        return (
            single_or_sweep_values(args.goal_x, args.goal_x_values),
            single_or_sweep_values(args.goal_y, args.goal_y_values),
            single_or_sweep_values(args.goal_z, args.goal_z_values),
        )

    offsets = [-args.goal_grid_spacing, 0.0, args.goal_grid_spacing]
    goal_x_values = [args.goal_grid_center_x + offset for offset in offsets]
    goal_y_values = [args.goal_grid_center_y + offset for offset in offsets]
    goal_z_values = single_or_sweep_values(args.goal_z, args.goal_z_values)
    if goal_z_values == [None]:
        goal_z_values = [args.goal_grid_z]
    return goal_x_values, goal_y_values, goal_z_values


def make_initial_state_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    cube_x_values = single_or_sweep_values(args.cube_x, args.cube_x_values)
    cube_y_values = single_or_sweep_values(args.cube_y, args.cube_y_values)
    cube_yaw_values = single_or_sweep_values(args.cube_yaw, args.cube_yaw_values)
    goal_x_values, goal_y_values, goal_z_values = goal_position_values(args)
    robot_offsets = parse_robot_qpos_offsets(args.robot_qpos_offsets)

    configs: list[dict[str, Any]] = []
    for robot_offset in robot_offsets:
        for cube_x in cube_x_values:
            for cube_y in cube_y_values:
                for cube_yaw in cube_yaw_values:
                    for goal_x in goal_x_values:
                        for goal_y in goal_y_values:
                            for goal_z in goal_z_values:
                                configs.append(
                                    dict(
                                        robot_qpos_offset=robot_offset,
                                        cube_x=cube_x,
                                        cube_y=cube_y,
                                        cube_yaw=cube_yaw,
                                        goal_x=goal_x,
                                        goal_y=goal_y,
                                        goal_z=goal_z,
                                    )
                                )
    return configs


def apply_initial_state_config(env, config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}

    base_env = env.unwrapped
    applied: dict[str, Any] = {}

    robot_offset = np.asarray(config.get("robot_qpos_offset", []), dtype=np.float32)
    if robot_offset.size:
        qpos = base_env.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
        if robot_offset.size == 7:
            qpos[:7] += robot_offset
        elif robot_offset.size == qpos.size:
            qpos += robot_offset
        else:
            raise ValueError(
                f"Robot qpos offset has size {robot_offset.size}, expected 7 or {qpos.size}."
            )
        base_env.agent.robot.set_qpos(qpos)
        base_env.agent.robot.set_qvel(np.zeros_like(qpos))
        applied["robot_qpos_offset"] = robot_offset.tolist()
        applied["robot_qpos"] = qpos.tolist()

    cube_x = config.get("cube_x")
    cube_y = config.get("cube_y")
    cube_yaw = config.get("cube_yaw")
    if cube_x is not None or cube_y is not None or cube_yaw is not None:
        cube_pose = base_env.cube.pose.sp
        cube_p = np.asarray(cube_pose.p, dtype=np.float32).copy()
        cube_q = np.asarray(cube_pose.q, dtype=np.float32).copy()
        if cube_x is not None:
            cube_p[0] = float(cube_x)
        if cube_y is not None:
            cube_p[1] = float(cube_y)
        if cube_yaw is not None:
            half = float(cube_yaw) * 0.5
            cube_q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)
        base_env.cube.set_pose(sapien.Pose(cube_p, cube_q))
        applied["cube_pose"] = np.concatenate([cube_p, cube_q]).tolist()

    goal_x = config.get("goal_x")
    goal_y = config.get("goal_y")
    goal_z = config.get("goal_z")
    if goal_x is not None or goal_y is not None or goal_z is not None:
        goal_pose = base_env.goal_site.pose.sp
        goal_p = np.asarray(goal_pose.p, dtype=np.float32).copy()
        goal_q = np.asarray(goal_pose.q, dtype=np.float32).copy()
        if goal_x is not None:
            goal_p[0] = float(goal_x)
        if goal_y is not None:
            goal_p[1] = float(goal_y)
        if goal_z is not None:
            goal_p[2] = float(goal_z)
        base_env.goal_site.set_pose(sapien.Pose(goal_p, goal_q))
        applied["goal_pose"] = np.concatenate([goal_p, goal_q]).tolist()

        if hasattr(base_env, "goal_box"):
            if getattr(base_env, "goal_box_follow_goal", True):
                base_env.goal_box.set_pose(sapien.Pose([goal_p[0], goal_p[1], 0.0]))
                applied["goal_box_pose"] = [float(goal_p[0]), float(goal_p[1]), 0.0]
            else:
                box_pose = base_env.goal_box.pose.sp
                applied["goal_box_pose"] = np.asarray(box_pose.p, dtype=np.float32).tolist()

    return applied


def is_done(terminated: Any, truncated: Any) -> bool:
    return _as_bool(terminated) or _as_bool(truncated)


def normalize_rgb_frame(frame: Any) -> np.ndarray | None:
    frame = to_numpy_tree(frame)
    if not isinstance(frame, np.ndarray):
        return None
    if frame.ndim < 3 or frame.shape[-1] not in (3, 4):
        return None

    while frame.ndim > 3:
        frame = frame[0]

    if frame.shape[0] < 8 or frame.shape[1] < 8:
        return None

    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def rgb_frames_by_camera(obs: Any) -> dict[str, np.ndarray]:
    obs = to_numpy_tree(obs)
    frames: dict[str, np.ndarray] = {}

    if not isinstance(obs, dict):
        return frames

    sensor_data = obs.get("sensor_data")
    if not isinstance(sensor_data, dict):
        return frames

    for camera_name, camera_data in sensor_data.items():
        if isinstance(camera_data, dict) and "rgb" in camera_data:
            frame = normalize_rgb_frame(camera_data["rgb"])
            if frame is not None:
                frames[str(camera_name)] = frame

    return frames


def save_video(frames: list[np.ndarray], output_path: str | Path, fps: int = 20) -> Path | None:
    if not frames:
        return None

    try:
        import imageio.v3 as iio
    except ImportError:
        print("imageio is not installed; skipping video save.")
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.asarray(frames), fps=fps)
    return path


def save_rgb_frames(
    frames_by_camera: dict[str, list[np.ndarray]],
    output_dir: str | Path,
    episode_id: int,
) -> list[Path]:
    try:
        import imageio.v3 as iio
    except ImportError:
        print("imageio is not installed; skipping RGB frame save.")
        return []

    paths: list[Path] = []
    for camera_name, frames in frames_by_camera.items():
        frame_dir = Path(output_dir) / "rgb_frames" / camera_name / f"episode_{episode_id:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        for frame_id, frame in enumerate(frames):
            path = frame_dir / f"frame_{frame_id:06d}.png"
            iio.imwrite(path, frame)
            paths.append(path)

    return paths


def json_safe(value: Any) -> Any:
    value = to_numpy_tree(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [json_safe(v) for v in value.reshape(-1).tolist()]
        if value.size <= 64:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def make_json_serializable(value: Any) -> Any:
    value = to_numpy_tree(value)

    if isinstance(value, dict):
        return {str(k): make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [make_json_serializable(v) for v in value.reshape(-1).tolist()]
        if value.size == 1:
            scalar = value.reshape(-1)[0]
            if isinstance(scalar, np.generic):
                return scalar.item()
            return make_json_serializable(scalar)
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def episode_data_path(output_dir: str | Path, episode_id: int) -> Path:
    return Path(output_dir) / "episodes" / f"episode_{episode_id:06d}.npz"


def save_episode_data(
    output_dir: str | Path,
    episode_id: int,
    observations: list[Any],
    actions: list[Any],
    rewards: list[float],
    dones: list[bool],
    infos: list[Any],
    metadata: dict[str, Any],
) -> Path:
    output_path = episode_data_path(output_dir, episode_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        episode_id=np.asarray(episode_id),
        num_steps=np.asarray(len(actions)),
        observations=np.asarray([to_numpy_tree(obs) for obs in observations], dtype=object),
        actions=np.asarray([to_numpy_tree(action) for action in actions], dtype=object),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        infos=np.asarray([to_numpy_tree(info) for info in infos], dtype=object),
        metadata=np.asarray(json.dumps(json_safe(metadata), ensure_ascii=False)),
    )

    return output_path


def first_scalar_bool_from_info(info: Any, keys: tuple[str, ...] = ("success", "is_success")) -> bool | None:
    info = to_numpy_tree(info)
    if not isinstance(info, dict):
        return None

    for key in keys:
        if key in info:
            return _as_bool(info[key])

    return None


def parse_pickcube_state(obs: Any) -> dict[str, np.ndarray | bool]:
    """Parse PickCube rgb+state observation.

    State layout:
      qpos 9, qvel 9, is_grasped 1, tcp_pose 7, goal_pos 3,
      obj_pose 7, tcp_to_obj_pos 3, obj_to_goal_pos 3.
    """
    obs_np = to_numpy_tree(obs)
    if not isinstance(obs_np, dict) or "state" not in obs_np:
        raise ValueError("Expected obs_mode='rgb+state' with obs['state'].")

    state = np.asarray(obs_np["state"], dtype=np.float32).reshape(-1)

    if state.shape[0] < 42:
        raise ValueError(f"Expected PickCube state dim >= 42, got {state.shape[0]}.")

    return {
        "qpos": state[0:9],
        "qvel": state[9:18],
        "is_grasped": bool(state[18] > 0.5),
        "tcp_pose": state[19:26],
        "goal_pos": state[26:29],
        "obj_pose": state[29:36],
        "tcp_to_obj_pos": state[36:39],
        "obj_to_goal_pos": state[39:42],
    }


class RecorderEnv:
    """Wrap an env so planner-internal env.step(action) calls are recorded."""

    def __init__(self, env) -> None:
        self.env = env
        self.observations: list[Any] = []
        self.actions: list[Any] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.infos: list[Any] = []
        self.frames_by_camera: dict[str, list[np.ndarray]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, *args, **kwargs):
        self.clear()
        obs, info = self.env.reset(*args, **kwargs)
        self.record_observation(obs, info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = is_done(terminated, truncated)

        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(scalar_float(reward))
        self.dones.append(done)
        self.infos.append(info)

        for camera_name, frame in rgb_frames_by_camera(obs).items():
            self.frames_by_camera.setdefault(camera_name, []).append(frame)

        return obs, reward, terminated, truncated, info

    def record_observation(self, obs: Any, info: Any) -> None:
        self.observations.append(obs)
        self.infos.append(info)

        for camera_name, frame in rgb_frames_by_camera(obs).items():
            self.frames_by_camera.setdefault(camera_name, []).append(frame)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.infos.clear()
        self.frames_by_camera.clear()

    def close(self) -> None:
        self.env.close()


def build_env(args: argparse.Namespace):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import robust_recovery.envs.pickcube_multicam  # noqa: F401

    kwargs: dict[str, Any] = dict(
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        max_episode_steps=args.max_steps,
        render_mode=args.render_mode,
        robot_init_qpos_noise=args.robot_init_qpos_noise,
    )

    if args.robot_uids:
        kwargs["robot_uids"] = args.robot_uids

    if args.image_width and args.image_height:
        kwargs["sensor_configs"] = dict(
            width=args.image_width,
            height=args.image_height,
        )

    return gym.make(args.env_id, **kwargs)


def print_observation_debug(obs: Any) -> None:
    obs_np = to_numpy_tree(obs)
    print("obs type:", type(obs).__name__)

    if isinstance(obs_np, dict):
        print("obs keys:", obs_np.keys())

        if "state" in obs_np:
            print("state shape:", np.asarray(obs_np["state"]).shape)

        sensor_data = obs_np.get("sensor_data")
        if isinstance(sensor_data, dict):
            print("sensor_data keys:", sensor_data.keys())
            for camera_name, camera_data in sensor_data.items():
                if isinstance(camera_data, dict) and "rgb" in camera_data:
                    frame = normalize_rgb_frame(camera_data["rgb"])
                    if frame is not None:
                        print(f"camera {camera_name}: rgb shape={frame.shape}, dtype={frame.dtype}")
    else:
        print("obs shape:", getattr(obs_np, "shape", None))


def planner_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "__dict__"):
        return make_json_serializable(result.__dict__)
    if isinstance(result, bool):
        return {"success": result}
    return {"result": make_json_serializable(result)}


def run_episode(
    env,
    args: argparse.Namespace,
    episode_id: int,
    initial_state_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recorder = RecorderEnv(env)
    obs, info = recorder.reset(seed=args.seed + episode_id)
    applied_initial_state = apply_initial_state_config(recorder, initial_state_config)
    if applied_initial_state:
        info = recorder.unwrapped.get_info()
        obs = recorder.unwrapped.get_obs(info)
        recorder.clear()
        recorder.record_observation(obs, info)

    if episode_id == 0:
        print_observation_debug(obs)

    parsed = parse_pickcube_state(obs)
    obj_pos = np.asarray(parsed["obj_pose"][:3], dtype=np.float32)
    goal_pos = np.asarray(parsed["goal_pos"], dtype=np.float32)

    home_qpos = parse_optional_home_qpos(args.home_qpos)
    planner = PandaPickPlacePlanner(
        recorder,
        debug=args.debug_planner,
        vis=args.vis_planner,
        print_env_info=args.print_env_info,
        joint_vel_limits=args.planner_joint_vel_limits,
        joint_acc_limits=args.planner_joint_acc_limits,
        prefer_screw=not args.prefer_rrt,
        grasp_diversity=args.enable_grasp_diversity and not args.disable_grasp_diversity,
        grasp_candidate_count=args.grasp_candidate_count,
        refine_scale=args.refine_scale,
        rotate_on_approach=args.enable_grasp_diversity and not args.disable_grasp_diversity,
        rng_seed=args.seed + episode_id,
        place_height=args.planner_place_height,
        return_home=args.return_home,
        home_qpos=home_qpos,
    )

    planner_result = planner.pick_and_place(obj_pos=obj_pos, goal_pos=goal_pos)
    planner.close()

    detected_cameras = sorted(recorder.frames_by_camera.keys())
    missing_cameras = [name for name in args.expected_cameras if name not in detected_cameras]
    planner_result_dict = planner_result_to_dict(planner_result)
    final_info = recorder.infos[-1] if recorder.infos else {}

    metadata = dict(
        env_id=args.env_id,
        robot_uids=args.robot_uids,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        max_steps=args.max_steps,
        seed=args.seed + episode_id,
        sweep_index=episode_id,
        initial_state_config=applied_initial_state,
        policy="motionplanning",
        planner="PandaPickPlacePlanner",
        planner_joint_vel_limits=args.planner_joint_vel_limits,
        planner_joint_acc_limits=args.planner_joint_acc_limits,
        planner_prefer_screw=not args.prefer_rrt,
        planner_grasp_diversity=args.enable_grasp_diversity and not args.disable_grasp_diversity,
        planner_grasp_candidate_count=args.grasp_candidate_count,
        planner_refine_scale=args.refine_scale,
        planner_rotate_on_approach=args.enable_grasp_diversity and not args.disable_grasp_diversity,
        planner_place_height=args.planner_place_height,
        planner_return_home=args.return_home,
        planner_home_qpos=home_qpos,
        goal_grid_3x3=args.goal_grid_3x3,
        goal_grid_center=[args.goal_grid_center_x, args.goal_grid_center_y],
        goal_grid_spacing=args.goal_grid_spacing,
        goal_grid_z=args.goal_grid_z,
        obj_pos_initial=obj_pos,
        goal_pos_initial=goal_pos,
        detected_cameras=detected_cameras,
        expected_cameras=args.expected_cameras,
        missing_cameras=missing_cameras,
        image_width=args.image_width,
        image_height=args.image_height,
        env_success=first_scalar_bool_from_info(final_info),
        planner_result=planner_result_dict,
    )

    data_path = save_episode_data(
        args.output_dir,
        episode_id,
        recorder.observations,
        recorder.actions,
        recorder.rewards,
        recorder.dones,
        recorder.infos,
        metadata,
    )

    frame_paths: list[Path] = []
    if args.save_rgb_frames:
        frame_paths = save_rgb_frames(recorder.frames_by_camera, args.output_dir, episode_id)

    video_paths: list[str] = []
    if args.save_videos:
        for camera_name, frames in recorder.frames_by_camera.items():
            video_path = Path(args.output_dir) / "videos" / camera_name / f"episode_{episode_id:06d}.mp4"
            saved = save_video(frames, video_path, fps=args.video_fps)
            if saved is not None:
                video_paths.append(str(saved))

    if missing_cameras:
        print(
            "Warning: expected cameras not found in observation:",
            missing_cameras,
            "Detected cameras:",
            detected_cameras,
        )

    return dict(
        episode_id=episode_id,
        data_path=str(data_path),
        initial_state_config=applied_initial_state,
        num_steps=len(recorder.actions),
        total_reward=float(np.sum(recorder.rewards)) if recorder.rewards else 0.0,
        env_success=metadata["env_success"],
        planner_result=planner_result_dict,
        detected_cameras=detected_cameras,
        missing_cameras=missing_cameras,
        num_saved_frames=len(frame_paths),
        video_paths=video_paths,
    )


def save_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    path = output_dir / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_serializable(summaries), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickCubeMultiCam-v1")
    parser.add_argument("--robot-uids", default="panda")
    parser.add_argument("--obs-mode", default="rgb+state")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--reward-mode", default="normalized_dense")
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot-init-qpos-noise", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/nominal/pickcube_motionplanning_multicam"),
    )
    parser.add_argument("--save-rgb-frames", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--debug-planner", action="store_true")
    parser.add_argument("--vis-planner", action="store_true")
    parser.add_argument("--print-env-info", action="store_true")
    parser.add_argument("--planner-joint-vel-limits", type=float, default=0.55)
    parser.add_argument("--planner-joint-acc-limits", type=float, default=0.45)
    parser.add_argument("--refine-scale", type=int, default=1)
    parser.add_argument("--planner-place-height", type=float)
    parser.add_argument("--return-home", dest="return_home", action="store_true", default=True)
    parser.add_argument("--no-return-home", dest="return_home", action="store_false")
    parser.add_argument(
        "--home-qpos",
        help="Comma-separated 7-DoF Panda arm qpos to return to at the end.",
    )
    parser.add_argument("--grasp-candidate-count", type=int, default=4)
    parser.add_argument("--enable-grasp-diversity", action="store_true")
    parser.add_argument("--disable-grasp-diversity", action="store_true")
    parser.add_argument("--prefer-rrt", action="store_true")
    parser.add_argument("--cube-x", type=float)
    parser.add_argument("--cube-y", type=float)
    parser.add_argument("--cube-yaw", type=float)
    parser.add_argument("--cube-x-values", nargs="*", type=float)
    parser.add_argument("--cube-y-values", nargs="*", type=float)
    parser.add_argument("--cube-yaw-values", nargs="*", type=float)
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--goal-z", type=float)
    parser.add_argument("--goal-x-values", nargs="*", type=float)
    parser.add_argument("--goal-y-values", nargs="*", type=float)
    parser.add_argument("--goal-z-values", nargs="*", type=float)
    parser.add_argument("--goal-grid-3x3", action="store_true")
    parser.add_argument("--goal-grid-center-x", type=float, default=0.03)
    parser.add_argument("--goal-grid-center-y", type=float, default=0.0)
    parser.add_argument("--goal-grid-spacing", type=float, default=0.046)
    parser.add_argument("--goal-grid-z", type=float, default=0.024)
    parser.add_argument(
        "--robot-qpos-offsets",
        nargs="*",
        help=(
            "Comma-separated qpos offsets. Use 7 values for arm joints or 9 values "
            "for full Panda qpos, e.g. '0,0.1,0,0,0,0,0'."
        ),
    )
    parser.add_argument(
        "--expected-cameras",
        nargs="*",
        default=["base_camera", "top_camera", "side_camera", "wrist_camera"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_configs = make_initial_state_grid(args)
    if len(sweep_configs) > 1 and args.episodes == 1:
        args.episodes = len(sweep_configs)

    env = build_env(args)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    summaries: list[dict[str, Any]] = []
    try:
        for episode_id in trange(args.episodes, desc="Collecting nominal episodes"):
            initial_state_config = sweep_configs[episode_id % len(sweep_configs)]
            summary = run_episode(env, args, episode_id, initial_state_config)
            summaries.append(summary)
            print(f"saved {summary['data_path']}")
    finally:
        env.close()

    summary_path = save_summary(args.output_dir, summaries)
    print(f"summary saved to {summary_path}")


if __name__ == "__main__":
    main()
