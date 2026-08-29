"""Create idle-frame filtered copies of saved nominal datasets.

The source episodes have one initial observation followed by one observation per
action. Filtered episodes intentionally store one observation per kept action:
the observation is the post-action frame, so image/frame/action rows stay aligned.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import imageio.v2 as imageio
import numpy as np


CAMERAS = ("base_camera", "top_camera", "side_camera", "left_wrist_camera", "right_wrist_camera")


@dataclass(frozen=True)
class FilterPreset:
    name: str
    action_threshold: float
    image_threshold: float
    cumulative_image_threshold: float
    gripper_threshold: float
    max_skip: int
    protect_window: int


PRESETS = {
    "train": FilterPreset(
        name="filtered_train",
        action_threshold=0.0035,
        image_threshold=0.35,
        cumulative_image_threshold=1.25,
        gripper_threshold=0.15,
        max_skip=6,
        protect_window=3,
    ),
    "demo": FilterPreset(
        name="filtered_demo",
        action_threshold=0.020,
        image_threshold=1.20,
        cumulative_image_threshold=3.50,
        gripper_threshold=0.15,
        max_skip=8,
        protect_window=2,
    ),
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [json_safe(v) for v in value.reshape(-1).tolist()]
        if value.size == 1:
            return json_safe(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    raw = data["metadata"].item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    return {}


def action_to_vector(action: Any) -> np.ndarray:
    if isinstance(action, dict):
        flat: list[float] = []
        for key in sorted(action):
            flat.extend(np.asarray(action[key], dtype=np.float32).reshape(-1).tolist())
        return np.asarray(flat, dtype=np.float32)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def gripper_values(action: Any) -> np.ndarray:
    if isinstance(action, dict):
        values = []
        for key in sorted(action):
            vec = np.asarray(action[key], dtype=np.float32).reshape(-1)
            if vec.size:
                values.append(float(vec[-1]))
        return np.asarray(values, dtype=np.float32)
    vec = np.asarray(action, dtype=np.float32).reshape(-1)
    if vec.size >= 16:
        return np.asarray([vec[7], vec[15]], dtype=np.float32)
    if vec.size:
        return np.asarray([vec[-1]], dtype=np.float32)
    return np.zeros((0,), dtype=np.float32)


def camera_rgb(obs: Any, camera: str) -> np.ndarray | None:
    if not isinstance(obs, dict):
        return None
    sensor_data = obs.get("sensor_data")
    if not isinstance(sensor_data, dict) or camera not in sensor_data:
        return None
    camera_data = sensor_data[camera]
    if not isinstance(camera_data, dict) or "rgb" not in camera_data:
        return None
    frame = np.asarray(camera_data["rgb"])
    while frame.ndim > 3:
        frame = frame[0]
    if frame.ndim < 3 or frame.shape[-1] < 3:
        return None
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def image_diff(prev_obs: Any, cur_obs: Any, cameras: tuple[str, ...]) -> float:
    diffs = []
    for camera in cameras:
        a = camera_rgb(prev_obs, camera)
        b = camera_rgb(cur_obs, camera)
        if a is None or b is None or a.shape != b.shape:
            continue
        diffs.append(float(np.mean(np.abs(b.astype(np.int16) - a.astype(np.int16)))))
    return max(diffs) if diffs else 0.0


def choose_action_indices(
    observations: np.ndarray,
    actions: np.ndarray,
    preset: FilterPreset,
    cameras: tuple[str, ...],
) -> tuple[list[int], dict[str, Any]]:
    action_vecs = [action_to_vector(action) for action in actions]
    grippers = [gripper_values(action) for action in actions]
    action_deltas = np.zeros(len(actions), dtype=np.float32)
    gripper_deltas = np.zeros(len(actions), dtype=np.float32)
    image_diffs = np.zeros(len(actions), dtype=np.float32)

    for i in range(len(actions)):
        if i > 0:
            action_deltas[i] = float(np.linalg.norm(action_vecs[i] - action_vecs[i - 1]))
            if grippers[i].shape == grippers[i - 1].shape:
                gripper_deltas[i] = float(np.max(np.abs(grippers[i] - grippers[i - 1]))) if grippers[i].size else 0.0
        # Action i leads to observation i + 1. Compare the visible result with
        # the preceding observation so we do not delete visually meaningful frames.
        if i + 1 < len(observations):
            image_diffs[i] = image_diff(observations[i], observations[i + 1], cameras)

    forced = np.zeros(len(actions), dtype=bool)
    for i in range(len(actions)):
        if i == 0 or i == len(actions) - 1:
            forced[i] = True
        if gripper_deltas[i] >= preset.gripper_threshold:
            lo = max(0, i - preset.protect_window)
            hi = min(len(actions), i + preset.protect_window + 1)
            forced[lo:hi] = True

    keep: list[int] = []
    last_kept = -1
    last_kept_obs = observations[1] if len(observations) > 1 else observations[0]
    reasons = {
        "forced": 0,
        "action": 0,
        "image": 0,
        "cumulative_image": 0,
        "max_skip": 0,
    }

    for i in range(len(actions)):
        cumulative_diff = image_diff(last_kept_obs, observations[i + 1], cameras) if i + 1 < len(observations) else 0.0
        reason = None
        if forced[i]:
            reason = "forced"
        elif action_deltas[i] >= preset.action_threshold:
            reason = "action"
        elif image_diffs[i] >= preset.image_threshold:
            reason = "image"
        elif cumulative_diff >= preset.cumulative_image_threshold:
            reason = "cumulative_image"
        elif last_kept < 0 or i - last_kept >= preset.max_skip:
            reason = "max_skip"

        if reason is not None:
            keep.append(i)
            last_kept = i
            if i + 1 < len(observations):
                last_kept_obs = observations[i + 1]
            reasons[reason] += 1

    if keep[-1] != len(actions) - 1:
        keep.append(len(actions) - 1)
        reasons["forced"] += 1

    stats = {
        "preset": preset.name,
        "source_action_count": int(len(actions)),
        "kept_action_count": int(len(keep)),
        "removed_action_count": int(len(actions) - len(keep)),
        "kept_fraction": float(len(keep) / max(1, len(actions))),
        "action_threshold": preset.action_threshold,
        "image_threshold": preset.image_threshold,
        "cumulative_image_threshold": preset.cumulative_image_threshold,
        "max_skip": preset.max_skip,
        "protect_window": preset.protect_window,
        "reason_counts": reasons,
        "action_delta_mean_before": float(np.mean(action_deltas)) if action_deltas.size else 0.0,
        "image_diff_mean_before": float(np.mean(image_diffs)) if image_diffs.size else 0.0,
    }
    return keep, stats


def save_video_from_observations(observations: np.ndarray, output_dir: Path, episode_id: int, fps: int, cameras: tuple[str, ...]) -> list[str]:
    video_paths: list[str] = []
    for camera in cameras:
        frames = [camera_rgb(obs, camera) for obs in observations]
        valid = [frame for frame in frames if frame is not None]
        if not valid:
            continue
        path = output_dir / "videos" / camera / f"episode_{episode_id:06d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, np.asarray(valid, dtype=np.uint8), fps=fps)
        video_paths.append(str(path))
    return video_paths


def make_grid_frame(frames: list[np.ndarray], columns: int = 3) -> np.ndarray:
    if not frames:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    h = max(frame.shape[0] for frame in frames)
    w = max(frame.shape[1] for frame in frames)
    rows = int(np.ceil(len(frames) / columns))
    canvas = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r, c = divmod(idx, columns)
        canvas[r * h : r * h + frame.shape[0], c * w : c * w + frame.shape[1]] = frame
    return canvas


def save_combined_video(output_dir: Path, fps: int, cameras: tuple[str, ...]) -> None:
    episode_paths = sorted((output_dir / "episodes").glob("episode_*.npz"))
    if not episode_paths:
        return
    output_path = output_dir / "combined" / "combined_all_cameras.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for episode_path in episode_paths:
            data = np.load(episode_path, allow_pickle=True)
            observations = data["observations"]
            for obs in observations:
                frames = []
                for camera in cameras:
                    frame = camera_rgb(obs, camera)
                    if frame is not None:
                        frames.append(frame)
                writer.append_data(make_grid_frame(frames))


def filter_episode(source_npz: Path, output_dir: Path, episode_id: int, preset: FilterPreset, fps: int, cameras: tuple[str, ...]) -> dict[str, Any]:
    data = np.load(source_npz, allow_pickle=True)
    observations = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    infos = data["infos"]
    metadata = metadata_from_npz(data)

    keep, stats = choose_action_indices(observations, actions, preset, cameras)
    # Keep post-action observations so rows are action/frame aligned.
    obs_indices = [i + 1 for i in keep]
    filtered_observations = observations[obs_indices]
    filtered_actions = actions[keep]
    filtered_rewards = rewards[keep]
    filtered_dones = dones[keep]
    info_indices = [i + 1 for i in keep] if len(infos) == len(observations) else keep
    filtered_infos = infos[info_indices]

    metadata = dict(metadata)
    metadata["num_steps_original"] = int(len(actions))
    metadata["num_steps"] = int(len(filtered_actions))
    metadata["filter_info"] = {
        **stats,
        "source_episode": str(source_npz),
        "action_frame_alignment": "observations[k] is the post-action frame for actions[k]",
        "kept_action_indices": [int(i) for i in keep],
        "kept_observation_indices": [int(i) for i in obs_indices],
    }

    episode_path = output_dir / "episodes" / f"episode_{episode_id:06d}.npz"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        episode_path,
        episode_id=np.asarray(episode_id),
        num_steps=np.asarray(len(filtered_actions)),
        observations=np.asarray(filtered_observations, dtype=object),
        actions=np.asarray(filtered_actions, dtype=object),
        rewards=np.asarray(filtered_rewards, dtype=np.float32),
        dones=np.asarray(filtered_dones, dtype=bool),
        infos=np.asarray(filtered_infos, dtype=object),
        metadata=np.asarray(json.dumps(json_safe(metadata), ensure_ascii=False)),
    )
    video_paths = save_video_from_observations(filtered_observations, output_dir, episode_id, fps, cameras)
    final_error = metadata.get("obj_to_goal_pos_final")
    return {
        "episode_id": int(episode_id),
        "env_success": bool(metadata.get("env_success", True)),
        "data_path": str(episode_path),
        "video_paths": video_paths,
        "num_steps_original": int(len(actions)),
        "num_steps": int(len(filtered_actions)),
        "kept_fraction": stats["kept_fraction"],
        "obj_to_goal_pos_final": final_error,
    }


def copy_raw(source_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)


def write_reports(output_dir: Path, rows: list[dict[str, Any]], preset: FilterPreset) -> None:
    total = len(rows)
    successes = sum(1 for row in rows if row.get("env_success"))
    report = {
        "total": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": float(successes / total) if total else 0.0,
        "filter_preset": preset.name,
        "mean_kept_fraction": float(np.mean([row["kept_fraction"] for row in rows])) if rows else 0.0,
        "episodes": rows,
    }
    (output_dir / "success_report.json").write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False))
    summary = []
    for row in rows:
        summary.append(
            {
                "episode_id": row["episode_id"],
                "data_path": row["data_path"],
                "num_steps_original": row["num_steps_original"],
                "num_steps": row["num_steps"],
                "kept_fraction": row["kept_fraction"],
                "env_success": row["env_success"],
                "obj_to_goal_pos_final": row.get("obj_to_goal_pos_final"),
                "video_paths": row["video_paths"],
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--cameras", nargs="*", default=list(CAMERAS))
    parser.add_argument("--skip-raw-copy", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir
    output_prefix = args.output_prefix or source_dir.with_name(source_dir.name + "_idle_filtered")
    cameras = tuple(str(camera) for camera in args.cameras)

    if not args.skip_raw_copy:
        raw_dir = output_prefix.with_name(output_prefix.name + "_raw_copy")
        copy_raw(source_dir, raw_dir)
        print(f"raw copy saved to {raw_dir}")

    episode_paths = sorted((source_dir / "episodes").glob("episode_*.npz"))
    if not episode_paths:
        raise SystemExit(f"No episodes found under {source_dir / 'episodes'}")

    for preset_key in ("train", "demo"):
        preset = PRESETS[preset_key]
        output_dir = output_prefix.with_name(output_prefix.name + f"_{preset_key}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        rows = []
        for episode_id, episode_path in enumerate(episode_paths):
            row = filter_episode(episode_path, output_dir, episode_id, preset, args.video_fps, cameras)
            rows.append(row)
            print(
                f"{preset.name} episode {episode_id}: "
                f"{row['num_steps_original']} -> {row['num_steps']} "
                f"({row['kept_fraction'] * 100:.1f}%)"
            )
        write_reports(output_dir, rows, preset)
        save_combined_video(output_dir, args.video_fps, cameras)
        print(f"{preset.name} saved to {output_dir}")


if __name__ == "__main__":
    main()
