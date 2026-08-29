#!/usr/bin/env python3
"""Render a PI0.5 phone-slot episode as a single annotated demo video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_pi05_dataset_viewer import CAMERAS, build_frame_records


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3])
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)


def fmt_pose(pose: dict[str, Any]) -> str:
    pos = pose.get("position_xyz", [])
    quat = pose.get("quaternion_wxyz", [])
    return f"pos={pos} quat={quat}"


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        words = raw_line.split(" ")
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), trial, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        lines.append(current)
    return lines


def info_lines(record: dict[str, Any], instruction: str, slot_id: Any) -> list[str]:
    left = record.get("left_arm", {})
    right = record.get("right_arm", {})
    action = record.get("action", {})
    if isinstance(action, dict):
        action_text = json.dumps(action, ensure_ascii=False)
    else:
        action_text = str(action)
    if len(action_text) > 170:
        action_text = action_text[:167] + "..."
    return [
        f"Task: {instruction}",
        f"Frame: {record.get('frame')}    Slot ID: {slot_id}    Reward: {record.get('reward')}    Done: {record.get('done')}",
        f"Phone pose: {fmt_pose(record.get('phone', {}))}",
        f"Left arm TCP:  {fmt_pose(left.get('tcp', {}))}",
        f"Right arm TCP: {fmt_pose(right.get('tcp', {}))}",
        f"Left TCP -> phone: {left.get('tcp_to_phone', [])}",
        f"Right TCP -> phone: {right.get('tcp_to_phone', [])}",
        f"Goal pos: {record.get('goal_pos', [])}    Phone -> goal: {record.get('phone_to_goal', [])}",
        f"Action: {action_text}",
    ]


def draw_info_panel(
    width: int,
    height: int,
    record: dict[str, Any],
    instruction: str,
    slot_id: Any,
    font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin = 14
    y = 12
    draw.text((margin, y), "PI0.5 training fields per frame", fill=(20, 30, 45), font=title_font)
    y += 28
    max_width = width - margin * 2
    for line in info_lines(record, instruction, slot_id):
        fill = (0, 88, 168) if line.startswith("Task:") else (35, 45, 60)
        for wrapped in wrap_text(draw, line, font, max_width):
            draw.text((margin, y), wrapped, fill=fill, font=font)
            y += 19
            if y > height - 22:
                draw.text((margin, y), "...", fill=(35, 45, 60), font=font)
                return np.asarray(image, dtype=np.uint8)
        y += 2
    return np.asarray(image, dtype=np.uint8)


def video_iterators(dataset_dir: Path, episode_id: int) -> list[Iterable[np.ndarray]]:
    iters = []
    for camera in CAMERAS:
        path = dataset_dir / "videos" / camera / f"episode_{episode_id:06d}.mp4"
        if not path.exists():
            raise FileNotFoundError(path)
        iters.append(imageio.get_reader(path))
    return iters


def camera_label_bar(width: int, label: str, font: ImageFont.ImageFont) -> np.ndarray:
    bar = Image.new("RGB", (width, 24), (245, 247, 250))
    draw = ImageDraw.Draw(bar)
    draw.text((8, 5), label, fill=(45, 55, 70), font=font)
    return np.asarray(bar, dtype=np.uint8)


def stack_camera_grid(frames: list[np.ndarray], cell_size: tuple[int, int], font: ImageFont.ImageFont) -> np.ndarray:
    cells = []
    for camera, frame in zip(CAMERAS, frames):
        resized = resize_rgb(frame, cell_size)
        label = camera_label_bar(cell_size[0], camera, font)
        cells.append(np.concatenate([label, resized], axis=0))
    top = np.concatenate([cells[0], cells[1]], axis=1)
    bottom = np.concatenate([cells[2], cells[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def build_demo_video(
    dataset_dir: Path,
    episode_id: int,
    output: Path,
    fps: int,
    cell_size: tuple[int, int],
    panel_height: int,
    max_frames: int | None,
) -> None:
    npz_path = dataset_dir / "episodes" / f"episode_{episode_id:06d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    metadata, records = build_frame_records(npz_path)
    instruction = str(metadata.get("language_instruction", ""))
    slot_id = (metadata.get("initial_state_config") or {}).get("slot_id")
    readers = video_iterators(dataset_dir, episode_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    font = load_font(14)
    title_font = load_font(18)
    grid_width = cell_size[0] * 2
    panel_width = grid_width

    try:
        with imageio.get_writer(output, fps=fps, macro_block_size=1) as writer:
            frame_limit = min(len(records), max_frames if max_frames is not None else len(records))
            for frame_idx in range(frame_limit):
                camera_frames = []
                for reader in readers:
                    camera_frames.append(reader.get_data(frame_idx))
                grid = stack_camera_grid(camera_frames, cell_size, font)
                panel = draw_info_panel(
                    panel_width,
                    panel_height,
                    records[frame_idx],
                    instruction,
                    slot_id,
                    font,
                    title_font,
                )
                if panel.shape[1] != grid.shape[1]:
                    panel = resize_rgb(panel, (grid.shape[1], panel.shape[0]))
                writer.append_data(np.concatenate([grid, panel], axis=0))
    finally:
        for reader in readers:
            reader.close()


def parse_size(text: str) -> tuple[int, int]:
    if "x" not in text:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    width, height = text.lower().split("x", 1)
    return int(width), int(height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--cell-size", type=parse_size, default=(320, 320), help="Camera cell size as WIDTHxHEIGHT.")
    parser.add_argument("--panel-height", type=int, default=260)
    parser.add_argument("--max-frames", type=int, help="Debug option: render only the first N frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (
        args.dataset_dir / "demo_videos" / f"episode_{args.episode_id:06d}_pi05_demo.mp4"
    )
    build_demo_video(
        dataset_dir=args.dataset_dir,
        episode_id=args.episode_id,
        output=output,
        fps=args.fps,
        cell_size=args.cell_size,
        panel_height=args.panel_height,
        max_frames=args.max_frames,
    )
    print(f"demo video saved to {output}")


if __name__ == "__main__":
    main()
