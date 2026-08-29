#!/usr/bin/env python
"""Drop-test the PickCubeBox 3x3 tray collision with a dynamic object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sapien


def to_numpy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_numpy(item) for item in value]
    return value


def normalize_rgb_frame(frame: Any) -> np.ndarray | None:
    frame = to_numpy(frame)
    if not isinstance(frame, np.ndarray) or frame.ndim < 3 or frame.shape[-1] not in (3, 4):
        return None
    while frame.ndim > 3:
        frame = frame[0]
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def rgb_frames_by_camera(obs: Any) -> dict[str, np.ndarray]:
    obs = to_numpy(obs)
    frames: dict[str, np.ndarray] = {}
    sensor_data = obs.get("sensor_data") if isinstance(obs, dict) else None
    if not isinstance(sensor_data, dict):
        return frames
    for camera_name, camera_data in sensor_data.items():
        if isinstance(camera_data, dict) and "rgb" in camera_data:
            frame = normalize_rgb_frame(camera_data["rgb"])
            if frame is not None:
                frames[str(camera_name)] = frame
    return frames


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> str | None:
    if not frames:
        return None
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.asarray(frames), fps=fps)
    return str(path)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def actor_pose(actor) -> tuple[np.ndarray, np.ndarray]:
    pose = actor.pose.sp
    p = np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3]
    q = np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4]
    return p, q


def hold_robot_action(env) -> np.ndarray:
    base_env = env.unwrapped
    qpos = base_env.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
    if base_env.control_mode == "pd_joint_pos":
        return np.hstack([qpos[:7], 1.0]).astype(np.float32)
    if base_env.control_mode == "pd_joint_pos_vel":
        return np.hstack([qpos[:7], np.zeros(7, dtype=np.float32), 1.0]).astype(np.float32)
    return np.zeros(env.action_space.shape, dtype=np.float32)


def build_test_object(env, args: argparse.Namespace, position: np.ndarray):
    from mani_skill.utils.building import actors

    q = quat_from_euler(args.drop_roll, args.drop_pitch, args.drop_yaw)
    pose = sapien.Pose(position, q)
    if args.drop_shape == "sphere":
        return actors.build_sphere(
            env.unwrapped.scene,
            radius=args.sphere_radius,
            color=[1.0, 0.8, 0.05, 1.0],
            name="collision_test_sphere",
            body_type="dynamic",
            add_collision=True,
            initial_pose=pose,
        )

    return actors.build_cube(
        env.unwrapped.scene,
        half_size=args.cube_half_size,
        color=[1.0, 0.0, 0.0, 1.0],
        name="collision_test_cube",
        body_type="dynamic",
        add_collision=True,
        initial_pose=pose,
    )


def object_half_extent(args: argparse.Namespace) -> float:
    if args.drop_shape == "sphere":
        return args.sphere_radius
    return args.cube_half_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickCubeBoxMultiCam-v1")
    parser.add_argument("--robot-uids", default="panda")
    parser.add_argument("--obs-mode", default="rgb+state")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--reward-mode", default="normalized_dense")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--drop-shape", choices=["cube", "sphere"], default="cube")
    parser.add_argument("--cube-half-size", type=float, default=0.02)
    parser.add_argument("--sphere-radius", type=float, default=0.02)
    parser.add_argument("--target-x", type=float, default=0.03)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--drop-x", type=float, default=0.049)
    parser.add_argument("--drop-y", type=float, default=0.010)
    parser.add_argument("--drop-z", type=float, default=0.20)
    parser.add_argument("--drop-roll", type=float, default=0.35)
    parser.add_argument("--drop-pitch", type=float, default=0.18)
    parser.add_argument("--drop-yaw", type=float, default=0.60)
    parser.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/box_collision_drop"))
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import maniskill_insertion_demos.envs.pickcube_multicam  # noqa: F401

    args = parse_args()
    env = gym.make(
        args.env_id,
        robot_uids=args.robot_uids,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        max_episode_steps=max(args.steps + 5, 200),
        robot_init_qpos_noise=0.0,
    )

    frames_by_camera: dict[str, list[np.ndarray]] = {}
    poses: list[dict[str, list[float]]] = []
    try:
        obs, _ = env.reset(seed=args.seed)
        drop_position = np.array([args.drop_x, args.drop_y, args.drop_z], dtype=np.float32)
        test_object = build_test_object(env, args, drop_position)
        action = hold_robot_action(env)

        for frame in rgb_frames_by_camera(obs).items():
            frames_by_camera.setdefault(frame[0], []).append(frame[1])

        for _ in range(args.steps):
            obs, _, _, _, _ = env.step(action)
            p, q = actor_pose(test_object)
            poses.append({"p": p.astype(float).tolist(), "q": q.astype(float).tolist()})
            for camera_name, frame in rgb_frames_by_camera(obs).items():
                frames_by_camera.setdefault(camera_name, []).append(frame)

        final_pos = np.asarray(poses[-1]["p"], dtype=np.float32)
        final_quat = np.asarray(poses[-1]["q"], dtype=np.float32)
        base_env = env.unwrapped
        half_extent = object_half_extent(args)
        expected_floor_center_z = float(getattr(base_env, "box_floor_thickness", 0.004) + half_extent)
        cell_half = np.asarray(getattr(base_env, "box_inner_half_size", (0.021, 0.021)), dtype=np.float32)
        target_xy = np.array([args.target_x, args.target_y], dtype=np.float32)
        final_target_error = final_pos[:2] - target_xy
        drop_target_error = np.array([args.drop_x, args.drop_y], dtype=np.float32) - target_xy
        z_error = abs(float(final_pos[2]) - expected_floor_center_z)
        planar_tolerance = np.maximum(cell_half - half_extent, 0.0) + 0.003
        inside_target_cell = bool(np.all(np.abs(final_target_error) <= planar_tolerance))
        resting_on_floor = bool(z_error <= 0.006)

        video_paths: dict[str, str] = {}
        if args.save_videos:
            for camera_name, frames in frames_by_camera.items():
                video = save_video(
                    frames,
                    args.output_dir / "videos" / f"{camera_name}.mp4",
                    fps=args.video_fps,
                )
                if video is not None:
                    video_paths[camera_name] = video

        summary = {
            "env_id": args.env_id,
            "drop_shape": args.drop_shape,
            "cube_half_size": args.cube_half_size if args.drop_shape == "cube" else None,
            "sphere_radius": args.sphere_radius if args.drop_shape == "sphere" else None,
            "target_position_xy": [args.target_x, args.target_y],
            "drop_position": [args.drop_x, args.drop_y, args.drop_z],
            "drop_euler_rpy": [args.drop_roll, args.drop_pitch, args.drop_yaw],
            "drop_target_error_xy": drop_target_error.astype(float).tolist(),
            "final_position": final_pos.astype(float).tolist(),
            "final_quaternion_wxyz": final_quat.astype(float).tolist(),
            "final_target_error_xy": final_target_error.astype(float).tolist(),
            "expected_floor_center_z": expected_floor_center_z,
            "z_error": z_error,
            "inside_target_cell": inside_target_cell,
            "resting_on_floor": resting_on_floor,
            "collision_ok": bool(resting_on_floor),
            "poses": poses,
            "video_paths": video_paths,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "poses"}, indent=2))
        print(f"summary saved to {summary_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
