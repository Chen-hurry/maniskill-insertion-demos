"""Multi-camera PickCube environment for robust recovery experiments."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import sapien
from gymnasium.envs.registration import register

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig


def look_at(eye: list[float], target: list[float]) -> sapien.Pose:
    """Create a SAPIEN camera pose whose x-axis points from eye to target."""
    eye_np = np.asarray(eye, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)

    forward = target_np - eye_np
    forward = forward / np.linalg.norm(forward)

    left = np.cross([0.0, 0.0, 1.0], forward)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    left = left / np.linalg.norm(left)

    up = np.cross(forward, left)
    up = up / np.linalg.norm(up)

    mat44 = np.eye(4, dtype=np.float32)
    mat44[:3, 0] = forward
    mat44[:3, 1] = left
    mat44[:3, 2] = up
    mat44[:3, 3] = eye_np
    return sapien.Pose(mat44)


def build_open_box(
    scene,
    name: str,
    inner_half_size: tuple[float, float] = (0.021, 0.021),
    wall_thickness: float = 0.004,
    wall_height: float = 0.025,
    floor_thickness: float = 0.004,
    grid_shape: tuple[int, int] = (1, 1),
    color: tuple[float, float, float, float] = (0.1, 0.35, 1.0, 1.0),
    initial_pose: sapien.Pose | None = None,
):
    """Build an open-top collision box or grid tray centered on the table."""
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=color)
    cell_half_x, cell_half_y = inner_half_size
    rows, cols = grid_shape
    if rows < 1 or cols < 1:
        raise ValueError(f"grid_shape must be positive, got {grid_shape}")

    cell_x = cell_half_x * 2
    cell_y = cell_half_y * 2
    total_x = cols * cell_x + (cols + 1) * wall_thickness
    total_y = rows * cell_y + (rows + 1) * wall_thickness

    floor_half_size = [total_x / 2, total_y / 2, floor_thickness / 2]
    floor_pose = sapien.Pose([0, 0, floor_thickness / 2])
    builder.add_box_collision(floor_pose, floor_half_size)
    builder.add_box_visual(floor_pose, floor_half_size, material=material)

    wall_z = floor_thickness + wall_height / 2
    x_wall_half_size = [wall_thickness / 2, total_y / 2, wall_height / 2]
    for index in range(cols + 1):
        x = -total_x / 2 + wall_thickness / 2 + index * (cell_x + wall_thickness)
        pose = sapien.Pose([x, 0, wall_z])
        builder.add_box_collision(pose, x_wall_half_size)
        builder.add_box_visual(pose, x_wall_half_size, material=material)

    y_wall_half_size = [total_x / 2, wall_thickness / 2, wall_height / 2]
    for index in range(rows + 1):
        y = -total_y / 2 + wall_thickness / 2 + index * (cell_y + wall_thickness)
        pose = sapien.Pose([0, y, wall_z])
        builder.add_box_collision(pose, y_wall_half_size)
        builder.add_box_visual(pose, y_wall_half_size, material=material)

    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
    return builder.build_kinematic(name=name)


def _camera_config(uid: str, pose: sapien.Pose, width: int, height: int, **kwargs: Any) -> CameraConfig:
    signature = inspect.signature(CameraConfig)
    supported = set(signature.parameters.keys())

    params: dict[str, Any] = {
        "uid": uid,
        "pose": pose,
        "width": width,
        "height": height,
    }

    optional_defaults = {
        "fov": np.deg2rad(70),
        "near": 0.01,
        "far": 10.0,
    }
    for key, value in optional_defaults.items():
        if key in supported:
            params[key] = value

    for key, value in kwargs.items():
        if key in supported and value is not None:
            params[key] = value

    return CameraConfig(**params)


class PickCubeMultiCamEnv(PickCubeEnv):
    """PickCube with base/top/side fixed cameras and wrist camera."""

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        if hasattr(self, "goal_site"):
            self._hidden_objects = [
                obj for obj in self._hidden_objects if obj is not self.goal_site
            ]
            self.goal_site.show_visual()

    def _find_robot_link(self, names: list[str]):
        if not hasattr(self, "agent") or self.agent is None:
            return None

        for link in self.agent.robot.get_links():
            if link.name in names:
                return link
        return None

    @property
    def _default_sensor_configs(self):
        width = 256
        height = 256
        target = [0.0, 0.0, 0.05]

        cameras = [
            _camera_config(
                uid="base_camera",
                pose=look_at(eye=[0.45, -0.55, 0.45], target=target),
                width=width,
                height=height,
            ),
            _camera_config(
                uid="top_camera",
                pose=look_at(eye=[0.0, 0.0, 0.85], target=target),
                width=width,
                height=height,
            ),
            _camera_config(
                uid="side_camera",
                pose=look_at(eye=[-0.55, 0.35, 0.35], target=target),
                width=width,
                height=height,
            ),
        ]

        wrist_mount = self._find_robot_link(["panda_hand", "panda_link8", "panda_hand_tcp"])

        if wrist_mount is not None:
            cameras.append(
                _camera_config(
                    uid="wrist_camera",
                    pose=sapien.Pose(
                        p=[0.0, 0.0, 0.12],
                        q=[0.7071068, 0.0, -0.7071068, 0.0],
                    ),
                    width=width,
                    height=height,
                    fov=np.deg2rad(90),
                    mount=wrist_mount,
                )
            )
        else:
            print("Warning: could not find Panda wrist link; wrist_camera is disabled.")

        return cameras



class PickCubeBoxMultiCamEnv(PickCubeMultiCamEnv):
    """PickCube with fixed cameras and a visible 3x3 target tray."""

    box_center = (0.03, 0.0)
    box_floor_thickness = 0.004
    box_grid_shape = (3, 3)
    box_inner_half_size = (0.021, 0.021)
    box_wall_height = 0.025
    box_wall_thickness = 0.004
    goal_box_follow_goal = False

    @property
    def box_cell_pitch(self) -> tuple[float, float]:
        return (
            self.box_inner_half_size[0] * 2 + self.box_wall_thickness,
            self.box_inner_half_size[1] * 2 + self.box_wall_thickness,
        )

    def _load_scene(self, options: dict):
        self.goal_thresh = 0.012
        super()._load_scene(options)
        self.goal_box = build_open_box(
            self.scene,
            name="goal_box",
            inner_half_size=self.box_inner_half_size,
            wall_thickness=self.box_wall_thickness,
            wall_height=self.box_wall_height,
            floor_thickness=self.box_floor_thickness,
            grid_shape=self.box_grid_shape,
            initial_pose=sapien.Pose([self.box_center[0], self.box_center[1], 0.0]),
        )


def register_env() -> None:
    env_specs = [
        ("PickCubeMultiCam-v1", "robust_recovery.envs.pickcube_multicam:PickCubeMultiCamEnv"),
        ("PickCubeBoxMultiCam-v1", "robust_recovery.envs.pickcube_multicam:PickCubeBoxMultiCamEnv"),
    ]
    for env_id, entry_point in env_specs:
        try:
            register(
                id=env_id,
                entry_point=entry_point,
                max_episode_steps=100,
            )
        except Exception:
            pass


register_env()
