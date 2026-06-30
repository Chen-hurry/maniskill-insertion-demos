#!/usr/bin/env python
"""Smoke-test ManiSkill PickCube with Panda and save trajectory data/videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import trange


def to_numpy_tree(value: Any) -> Any:
    """Convert tensors and nested containers into numpy-friendly objects."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        return {key: to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [to_numpy_tree(item) for item in value]
        try:
            return np.asarray(converted)
        except ValueError:
            return converted
    if isinstance(value, (str, bytes)):
        return value
    return np.asarray(value)


def json_safe(value: Any) -> Any:
    value = to_numpy_tree(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_bool(value: Any) -> bool:
    array = to_numpy_tree(value)
    if isinstance(array, np.ndarray):
        return bool(array.any())
    return bool(array)


def is_done(terminated: Any, truncated: Any) -> bool:
    return _as_bool(terminated) or _as_bool(truncated)


def scalar_float(value: Any) -> float:
    array = to_numpy_tree(value)
    if isinstance(array, np.ndarray):
        return float(array.reshape(-1)[0])
    return float(array)


def first_scalar_bool_from_info(info: Any, keys: tuple[str, ...] = ("success", "is_success")) -> bool | None:
    info = to_numpy_tree(info)
    if not isinstance(info, dict):
        return None
    for key in keys:
        if key in info:
            return _as_bool(info[key])
    return None


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
    """Extract all RGB camera frames from a ManiSkill observation."""
    obs = to_numpy_tree(obs)
    frames: dict[str, np.ndarray] = {}

    if not isinstance(obs, dict):
        return frames

    sensor_data = obs.get("sensor_data")
    if isinstance(sensor_data, dict):
        for camera_name, camera_data in sensor_data.items():
            if not isinstance(camera_data, dict):
                continue
            if "rgb" in camera_data:
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
    )

    if args.robot_uids:
        kwargs["robot_uids"] = args.robot_uids

    if args.image_width and args.image_height:
        kwargs["sensor_configs"] = dict(width=args.image_width, height=args.image_height)

    try:
        return gym.make(args.env_id, **kwargs)
    except TypeError as exc:
        if "robot_uids" in kwargs:
            print(f"Warning: robot_uids={args.robot_uids!r} is not accepted by this env. Retrying without it.")
            kwargs.pop("robot_uids")
            return gym.make(args.env_id, **kwargs)
        raise exc


def print_observation_debug(obs: Any) -> None:
    obs_np = to_numpy_tree(obs)
    print("obs type:", type(obs).__name__)
    if isinstance(obs_np, dict):
        print("obs keys:", obs_np.keys())
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


def run_episode(
    env,
    args: argparse.Namespace,
    episode_id: int,
) -> dict[str, Any]:
    observations: list[Any] = []
    actions: list[Any] = []
    rewards: list[float] = []
    dones: list[bool] = []
    infos: list[Any] = []
    frames_by_camera: dict[str, list[np.ndarray]] = {}

    obs, info = env.reset(seed=args.seed + episode_id)

    if episode_id == 0:
        print_observation_debug(obs)

    observations.append(obs)
    infos.append(info)

    for camera_name, frame in rgb_frames_by_camera(obs).items():
        frames_by_camera.setdefault(camera_name, []).append(frame)

    for _ in range(getattr(env.unwrapped, "max_episode_steps", args.max_steps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = is_done(terminated, truncated)

        observations.append(obs)
        actions.append(action)
        rewards.append(scalar_float(reward))
        dones.append(done)
        infos.append(info)

        for camera_name, frame in rgb_frames_by_camera(obs).items():
            frames_by_camera.setdefault(camera_name, []).append(frame)

        if done:
            break

    detected_cameras = sorted(frames_by_camera.keys())
    missing_cameras = [name for name in args.expected_cameras if name not in detected_cameras]

    metadata = dict(
        env_id=args.env_id,
        robot_uids=args.robot_uids,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        max_steps=args.max_steps,
        seed=args.seed + episode_id,
        policy="random",
        detected_cameras=detected_cameras,
        expected_cameras=args.expected_cameras,
        missing_cameras=missing_cameras,
        image_width=args.image_width,
        image_height=args.image_height,
        success=first_scalar_bool_from_info(infos[-1]) if infos else None,
    )

    data_path = save_episode_data(
        args.output_dir,
        episode_id,
        observations,
        actions,
        rewards,
        dones,
        infos,
        metadata,
    )

    frame_paths: list[Path] = []
    if args.save_rgb_frames:
        frame_paths = save_rgb_frames(frames_by_camera, args.output_dir, episode_id)

    video_paths: list[str] = []
    if args.save_videos:
        for camera_name, frames in frames_by_camera.items():
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
        num_steps=len(actions),
        total_reward=float(np.sum(rewards)) if rewards else 0.0,
        success=metadata["success"],
        detected_cameras=detected_cameras,
        missing_cameras=missing_cameras,
        num_saved_frames=len(frame_paths),
        video_paths=video_paths,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--robot-uids", default="panda")
    parser.add_argument("--obs-mode", default="rgb+state")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--reward-mode", default="normalized_dense")
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--save-rgb-frames", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument(
        "--expected-cameras",
        nargs="*",
        default=["base_camera", "top_camera", "side_camera", "wrist_camera"],
    )
    return parser.parse_args()


def make_json_serializable(value: Any) -> Any:
    value = to_numpy_tree(value)

    if isinstance(value, dict):
        return {str(k): make_json_serializable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(v) for v in value]

    if isinstance(value, np.ndarray):
        # Object arrays may contain dict/list/path/array objects.
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


def save_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    path = output_dir / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_serializable(summaries), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    env = build_env(args)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    summaries: list[dict[str, Any]] = []
    try:
        for episode_id in trange(args.episodes, desc="Collecting episodes"):
            summary = run_episode(env, args, episode_id)
            summaries.append(summary)
            print(f"saved {summary['data_path']}")
    finally:
        env.close()

    summary_path = save_summary(args.output_dir, summaries)
    print(f"summary saved to {summary_path}")


if __name__ == "__main__":
    main()