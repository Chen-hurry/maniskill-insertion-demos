#!/usr/bin/env python3
"""Build a simple HTML viewer for one collected PI0.5 phone-slot episode."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("top_camera", "left_wrist_camera", "right_wrist_camera", "side_camera")
STATE_EXTRA_DIMS = {
    "left_arm_tcp": (0, 7),
    "right_arm_tcp": (7, 14),
    "obj_pose": (14, 21),
    "goal_pos": (21, 24),
    "left_tcp_to_obj_pos": (24, 27),
    "right_tcp_to_obj_pos": (27, 30),
    "obj_to_goal_pos": (30, 33),
}


def load_metadata(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    value = npz["metadata"]
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return jsonable(value.item())
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def as_state_vector(obs: Any) -> np.ndarray:
    if isinstance(obs, dict) and "state" in obs:
        state = np.asarray(obs["state"], dtype=np.float32)
        return state.reshape(-1)
    return np.asarray([], dtype=np.float32)


def pose_dict(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size >= 7:
        return {
            "position_xyz": [round(float(v), 5) for v in values[:3]],
            "quaternion_wxyz": [round(float(v), 5) for v in values[3:7]],
        }
    if values.size >= 3:
        return {"position_xyz": [round(float(v), 5) for v in values[:3]]}
    return {}


def extract_extra_state(state: np.ndarray) -> dict[str, Any]:
    # ManiSkill flattens agent state first, then env _get_obs_extra fields.
    extra_len = max(end for _, end in STATE_EXTRA_DIMS.values())
    if state.size < extra_len:
        return {"raw_state_head": [round(float(v), 5) for v in state[: min(16, state.size)]]}
    extra = state[-extra_len:]
    parsed: dict[str, Any] = {}
    for name, (start, end) in STATE_EXTRA_DIMS.items():
        values = extra[start:end]
        if name.endswith("_pose") or name.endswith("_tcp"):
            parsed[name] = pose_dict(values)
        else:
            parsed[name] = [round(float(v), 5) for v in values]
    return parsed


def action_summary(action: Any) -> Any:
    if isinstance(action, dict):
        result: dict[str, Any] = {}
        for key, value in action.items():
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            result[str(key)] = {
                "shape": list(np.asarray(value).shape),
                "values": [round(float(v), 5) for v in arr[: min(14, arr.size)]],
            }
        return result
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    return {
        "shape": list(np.asarray(action).shape),
        "values": [round(float(v), 5) for v in arr[: min(28, arr.size)]],
    }


def build_frame_records(npz_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = np.load(npz_path, allow_pickle=True)
    metadata = load_metadata(data)
    observations = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    infos = data["infos"]

    frame_count = int(len(observations))
    records: list[dict[str, Any]] = []
    for frame_idx in range(frame_count):
        obs = observations[frame_idx]
        state = as_state_vector(obs)
        record: dict[str, Any] = {
            "frame": frame_idx,
            "state_dim": int(state.size),
            "phone": extract_extra_state(state).get("obj_pose", {}),
            "left_arm": {
                "tcp": extract_extra_state(state).get("left_arm_tcp", {}),
                "tcp_to_phone": extract_extra_state(state).get("left_tcp_to_obj_pos", []),
            },
            "right_arm": {
                "tcp": extract_extra_state(state).get("right_arm_tcp", {}),
                "tcp_to_phone": extract_extra_state(state).get("right_tcp_to_obj_pos", []),
            },
            "goal_pos": extract_extra_state(state).get("goal_pos", []),
            "phone_to_goal": extract_extra_state(state).get("obj_to_goal_pos", []),
            "reward": round(float(rewards[frame_idx]), 5) if frame_idx < len(rewards) else None,
            "done": bool(dones[frame_idx]) if frame_idx < len(dones) else None,
            "action": action_summary(actions[frame_idx]) if frame_idx < len(actions) else None,
            "info": jsonable(infos[frame_idx]) if frame_idx < len(infos) else {},
        }
        records.append(record)
    return metadata, records


def rel_video_path(output_html: Path, dataset_dir: Path, camera: str, episode_id: int) -> str:
    video = dataset_dir / "videos" / camera / f"episode_{episode_id:06d}.mp4"
    return Path(os.path.relpath(video.resolve(), output_html.parent.resolve())).as_posix()


def render_html(
    dataset_dir: Path,
    episode_id: int,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    output_html: Path,
) -> str:
    instruction = metadata.get("language_instruction", "")
    env_success = metadata.get("env_success")
    slot_id = (metadata.get("initial_state_config") or {}).get("slot_id")
    initial_config = metadata.get("initial_state_config") or {}
    records_json = json.dumps(records, ensure_ascii=False)
    metadata_json = json.dumps(jsonable(metadata), ensure_ascii=False, indent=2)

    videos = "\n".join(
        f"""
        <section class="video-panel">
          <div class="panel-title">{html.escape(camera)}</div>
          <video id="video-{camera}" muted playsinline preload="metadata">
            <source src="{html.escape(rel_video_path(output_html, dataset_dir, camera, episode_id))}" type="video/mp4">
          </video>
        </section>
        """
        for camera in CAMERAS
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PI0.5 Phone Dataset Viewer - Episode {episode_id:06d}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5e6b7a;
      --line: #d9dee7;
      --accent: #0b6bcb;
      --ok: #177245;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 16px 20px 10px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 20px;
      font-weight: 650;
    }}
    .meta-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      font-size: 13px;
      color: var(--muted);
    }}
    .meta-line strong {{ color: var(--ink); }}
    main {{ padding: 16px 20px 22px; }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      max-width: 1180px;
    }}
    .video-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .panel-title {{
      padding: 8px 10px;
      font-size: 13px;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #111;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      max-width: 1180px;
      margin: 12px 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 11px;
      font-size: 13px;
      cursor: pointer;
    }}
    input[type="range"] {{ flex: 1; min-width: 180px; }}
    .frame-readout {{
      min-width: 118px;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
      font-size: 13px;
    }}
    .details {{
      max-width: 1180px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 15px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.45;
      color: #26313f;
    }}
    .instruction {{
      color: var(--accent);
      font-weight: 650;
    }}
    .success {{ color: var(--ok); }}
    @media (max-width: 760px) {{
      .video-grid, .details {{ grid-template-columns: 1fr; }}
      .controls {{ flex-wrap: wrap; }}
      input[type="range"] {{ width: 100%; flex-basis: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>PI0.5 Phone Dataset Viewer</h1>
    <div class="meta-line">
      <span>Episode <strong>{episode_id:06d}</strong></span>
      <span>Slot <strong>{html.escape(str(slot_id))}</strong></span>
      <span>Status <strong class="success">{html.escape(str(env_success))}</strong></span>
      <span>Frames <strong>{len(records)}</strong></span>
    </div>
    <div class="meta-line"><span>Instruction: <span class="instruction">{html.escape(str(instruction))}</span></span></div>
  </header>
  <main>
    <div class="video-grid">
      {videos}
    </div>
    <div class="controls">
      <button id="play">Play</button>
      <button id="pause">Pause</button>
      <button id="step-back">-1 frame</button>
      <button id="step-forward">+1 frame</button>
      <input id="frame-slider" type="range" min="0" max="{max(0, len(records) - 1)}" value="0">
      <div class="frame-readout" id="frame-readout">Frame 0</div>
    </div>
    <div class="details">
      <section class="card">
        <h2>Current Frame Signals</h2>
        <pre id="frame-json"></pre>
      </section>
      <section class="card">
        <h2>Episode Metadata</h2>
        <pre>{html.escape(metadata_json)}</pre>
      </section>
      <section class="card">
        <h2>Initial Configuration</h2>
        <pre>{html.escape(json.dumps(initial_config, ensure_ascii=False, indent=2))}</pre>
      </section>
      <section class="card">
        <h2>PI0.5 Fields</h2>
        <pre>{html.escape(json.dumps({
            "images": list(CAMERAS),
            "language_instruction": instruction,
            "state": "flattened robot/env state plus parsed phone and TCP poses",
            "action": "raw environment action; convert to 14D TCP pose + gripper for PI0.5 labels",
        }, ensure_ascii=False, indent=2))}</pre>
      </section>
    </div>
  </main>
  <script>
    const records = {records_json};
    const fps = 20;
    const videos = Array.from(document.querySelectorAll('video'));
    const slider = document.getElementById('frame-slider');
    const readout = document.getElementById('frame-readout');
    const frameJson = document.getElementById('frame-json');
    let syncing = false;

    function clampFrame(frame) {{
      return Math.max(0, Math.min(records.length - 1, frame));
    }}

    function showFrame(frame, seekVideos=true) {{
      frame = clampFrame(frame);
      slider.value = frame;
      readout.textContent = `Frame ${{frame}} / ${{records.length - 1}}`;
      frameJson.textContent = JSON.stringify(records[frame], null, 2);
      if (seekVideos) {{
        syncing = true;
        const t = frame / fps;
        for (const video of videos) video.currentTime = t;
        window.setTimeout(() => {{ syncing = false; }}, 80);
      }}
    }}

    function playAll() {{ for (const video of videos) video.play(); }}
    function pauseAll() {{ for (const video of videos) video.pause(); }}

    document.getElementById('play').addEventListener('click', playAll);
    document.getElementById('pause').addEventListener('click', pauseAll);
    document.getElementById('step-back').addEventListener('click', () => {{
      pauseAll();
      showFrame(Number(slider.value) - 1);
    }});
    document.getElementById('step-forward').addEventListener('click', () => {{
      pauseAll();
      showFrame(Number(slider.value) + 1);
    }});
    slider.addEventListener('input', () => {{
      pauseAll();
      showFrame(Number(slider.value));
    }});

    videos[0].addEventListener('timeupdate', () => {{
      if (syncing) return;
      const frame = clampFrame(Math.round(videos[0].currentTime * fps));
      showFrame(frame, false);
      const t = videos[0].currentTime;
      for (const video of videos.slice(1)) {{
        if (Math.abs(video.currentTime - t) > 0.08) video.currentTime = t;
        if (!videos[0].paused && video.paused) video.play();
      }}
    }});

    showFrame(0);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    npz_path = dataset_dir / "episodes" / f"episode_{args.episode_id:06d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    for camera in CAMERAS:
        video = dataset_dir / "videos" / camera / f"episode_{args.episode_id:06d}.mp4"
        if not video.exists():
            raise FileNotFoundError(video)

    output = args.output or (dataset_dir / f"viewer_episode_{args.episode_id:06d}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata, records = build_frame_records(npz_path)
    output.write_text(render_html(dataset_dir, args.episode_id, metadata, records, output), encoding="utf-8")
    print(f"viewer saved to {output}")


if __name__ == "__main__":
    main()
