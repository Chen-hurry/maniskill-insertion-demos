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


def register_env() -> None:
    try:
        register(
            id="PickCubeMultiCam-v1",
            entry_point="robust_recovery.envs.pickcube_multicam:PickCubeMultiCamEnv",
            max_episode_steps=100,
        )
    except Exception:
        pass


register_env()