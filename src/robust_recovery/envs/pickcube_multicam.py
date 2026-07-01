"""Multi-camera PickCube environment for robust recovery experiments."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import sapien
import torch
from gymnasium.envs.registration import register

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


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


def build_conveyor_platform(
    scene,
    name: str,
    half_size: tuple[float, float, float] = (0.12, 0.045, 0.010),
    color: tuple[float, float, float, float] = (0.18, 0.18, 0.18, 1.0),
    initial_pose: sapien.Pose | None = None,
):
    """Build a low conveyor-like support that presents the phone above the table."""
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=color)
    builder.add_box_collision(half_size=list(half_size))
    builder.add_box_visual(half_size=list(half_size), material=material)
    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
    return builder.build_kinematic(name=name)

def build_phone_actor(
    scene,
    name: str,
    half_size: tuple[float, float, float] = (0.075, 0.025, 0.0025),
    color: tuple[float, float, float, float] = (0.02, 0.02, 0.025, 1.0),
    initial_pose: sapien.Pose | None = None,
):
    """Build a phone-like cuboid with thickness:width:length around 1:10:30."""
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=color)
    builder.add_box_collision(half_size=list(half_size))
    builder.add_box_visual(half_size=list(half_size), material=material)

    # A tiny colored stripe marks the phone top/front in camera views.
    stripe_material = sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.85, 1.0])
    builder.add_box_visual(
        pose=sapien.Pose([half_size[0] * 0.55, 0, half_size[2] + 0.0006]),
        half_size=[half_size[0] * 0.12, half_size[1] * 0.75, 0.0005],
        material=stripe_material,
    )

    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
    return builder.build(name=name)


def build_phone_slot_tray(
    scene,
    name: str,
    slot_half_length: float = 0.004,
    slot_half_width: float = 0.028,
    slot_count: int = 3,
    wall_thickness: float = 0.004,
    wall_height: float = 0.090,
    floor_thickness: float = 0.004,
    color: tuple[float, float, float, float] = (0.05, 0.32, 1.0, 1.0),
    initial_pose: sapien.Pose | None = None,
):
    """Build open-ended side-wall slots for vertical phone insertion."""
    if slot_count < 1:
        raise ValueError(f"slot_count must be positive, got {slot_count}")

    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=color)

    slot_width = slot_half_width * 2
    total_x = slot_half_length * 2 + 2 * wall_thickness
    total_y = slot_count * slot_width + (slot_count - 1) * wall_thickness

    floor_pose = sapien.Pose([0, 0, floor_thickness / 2])
    floor_half_size = [total_x / 2, total_y / 2, floor_thickness / 2]
    builder.add_box_collision(floor_pose, floor_half_size)
    builder.add_box_visual(floor_pose, floor_half_size, material=material)

    wall_z = floor_thickness + wall_height / 2
    wall_half_size = [wall_thickness / 2, slot_half_width, wall_height / 2]
    for index in range(slot_count):
        y = (index - (slot_count - 1) / 2) * (slot_width + wall_thickness)
        for sign in (-1, 1):
            x = sign * (slot_half_length + wall_thickness / 2)
            pose = sapien.Pose([x, y, wall_z])
            builder.add_box_collision(pose, wall_half_size)
            builder.add_box_visual(pose, wall_half_size, material=material)

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


class PhoneSlotMultiCamEnv(PickCubeMultiCamEnv):
    """Phone-like cuboid pick-and-place task with a multi-slot groove tray."""

    phone_half_size = (0.075, 0.025, 0.0025)
    phone_spawn_center = (-0.08, 0.0)
    phone_spawn_half_size = 0.015
    conveyor_half_size = (0.12, 0.045, 0.010)
    conveyor_top_z = 0.020
    slot_center = (0.06, 0.0)
    slot_count = 3
    slot_floor_thickness = 0.004
    slot_half_length = 0.004
    slot_half_width = 0.028
    slot_wall_height = 0.090
    slot_wall_thickness = 0.004
    goal_box_follow_goal = False
    planner_grasp_z_offset = 0.000
    planner_default_place_height = 0.000
    planner_insert_angle_deg = 90.0
    planner_insert_rotation_q = (0.7071068, 0.0, -0.7071068, 0.0)
    planner_rotate_joint_index = 5
    planner_rotate_joint_delta = -np.pi / 2
    planner_use_single_joint_rotation = False
    planner_use_regularized_insert = True
    planner_joint_regularization_weights = (8.0, 8.0, 6.0, 4.0, 2.0, 1.0, 1.0)
    planner_regularized_terminal_weight = 1.0
    planner_regularized_smooth_weight = 0.25
    planner_regularized_duration_weight = 0.002
    planner_regularized_ik_samples = 32
    planner_regularized_ik_plan_candidates = 6
    planner_regularized_ik_threshold = 0.003
    planner_regularized_pose_attempts = 4
    planner_regularized_planning_time = 0.8
    planner_regularized_rrt_range = 0.08
    planner_two_panda_mode = "support"
    planner_handoff_angle_deg = 45.0
    planner_right_grasp_mode = "topdown_width"
    planner_left_receive_z_offset = 0.004
    planner_left_handoff_lift_height = 0.020
    planner_handoff_receive_mode = "topdown_center"
    planner_upper_side_receive_fraction = 0.08
    
    @property
    def slot_pitch(self) -> float:
        return self.slot_half_width * 2 + self.slot_wall_thickness

    @property
    def phone_goal_z(self) -> float:
        return self.slot_floor_thickness + self.phone_half_size[0]

    def _load_scene(self, options: dict):
        self.goal_thresh = 0.015
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.conveyor = build_conveyor_platform(
            self.scene,
            name="phone_conveyor",
            half_size=self.conveyor_half_size,
            initial_pose=sapien.Pose([self.phone_spawn_center[0], self.phone_spawn_center[1], self.conveyor_half_size[2]]),
        )
        self.cube = build_phone_actor(
            self.scene,
            name="phone",
            half_size=self.phone_half_size,
            initial_pose=sapien.Pose([0, 0, self.conveyor_top_z + self.phone_half_size[2]]),
        )
        from mani_skill.utils.building import actors

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self.goal_site.show_visual()
        self.goal_box = build_phone_slot_tray(
            self.scene,
            name="phone_slot_tray",
            slot_half_length=self.slot_half_length,
            slot_half_width=self.slot_half_width,
            slot_count=self.slot_count,
            wall_thickness=self.slot_wall_thickness,
            wall_height=self.slot_wall_height,
            floor_thickness=self.slot_floor_thickness,
            initial_pose=sapien.Pose([self.slot_center[0], self.slot_center[1], 0.0]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            phone_xyz = torch.zeros((b, 3))
            phone_xyz[:, :2] = (
                torch.rand((b, 2)) * self.phone_spawn_half_size * 2
                - self.phone_spawn_half_size
            )
            phone_xyz[:, 0] += self.phone_spawn_center[0]
            phone_xyz[:, 1] += self.phone_spawn_center[1]
            phone_xyz[:, 2] = self.conveyor_top_z + self.phone_half_size[2]
            phone_q = torch.zeros((b, 4))
            phone_q[:, 0] = 1.0
            self.cube.set_pose(Pose.create_from_pq(phone_xyz, phone_q))

            slot_offsets = torch.linspace(
                -(self.slot_count - 1) / 2,
                (self.slot_count - 1) / 2,
                self.slot_count,
            ) * self.slot_pitch
            slot_ids = torch.randint(0, self.slot_count, (b,))
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = self.slot_center[0]
            goal_xyz[:, 1] = torch.tensor(self.slot_center[1]) + slot_offsets[slot_ids]
            goal_xyz[:, 2] = self.phone_goal_z
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))


class TwoPandaPhoneSlotMultiCamEnv(PhoneSlotMultiCamEnv):
    """Two-Panda version of the phone-slot task.

    The right arm is intended to do the main pickup/insert motion while the
    left arm can move near the phone as a support/guard during rotation.
    """

    SUPPORTED_ROBOTS = [("panda", "panda"), ("panda_wristcam", "panda_wristcam")]

    def __init__(
        self,
        *args,
        robot_uids=("panda", "panda"),
        robot_init_qpos_noise=0.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        super(PickCubeEnv, self)._load_agent(
            options,
            [
                sapien.Pose(p=[0.0, -0.75, 0.0]),
                sapien.Pose(p=[0.0, 0.75, 0.0]),
            ],
        )

    @property
    def left_agent(self):
        return self.agent.agents[0]

    @property
    def right_agent(self):
        return self.agent.agents[1]

    def _find_agent_link(self, agent, names: list[str]):
        for link in agent.robot.get_links():
            if link.name in names:
                return link
        return None

    @property
    def _default_sensor_configs(self):
        width = 256
        height = 256
        target = [0.0, 0.0, 0.08]
        cameras = [
            _camera_config(
                uid="base_camera",
                pose=look_at(eye=[0.55, -0.65, 0.50], target=target),
                width=width,
                height=height,
            ),
            _camera_config(
                uid="top_camera",
                pose=look_at(eye=[0.0, 0.0, 0.95], target=target),
                width=width,
                height=height,
            ),
            _camera_config(
                uid="side_camera",
                pose=look_at(eye=[-0.65, 0.45, 0.42], target=target),
                width=width,
                height=height,
            ),
        ]

        if hasattr(self, "agent") and hasattr(self.agent, "agents"):
            for label, agent in (("left", self.left_agent), ("right", self.right_agent)):
                wrist_mount = self._find_agent_link(
                    agent,
                    ["panda_hand", "panda_link8", "panda_hand_tcp"],
                )
                if wrist_mount is not None:
                    cameras.append(
                        _camera_config(
                            uid=f"{label}_wrist_camera",
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
        return cameras

    def evaluate(self):
        obj_to_goal = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        is_obj_placed = obj_to_goal <= self.goal_thresh
        is_right_static = self.right_agent.is_static(0.2)
        is_left_static = self.left_agent.is_static(0.2)
        return {
            "success": torch.logical_and(is_obj_placed, torch.logical_and(is_right_static, is_left_static)),
            "is_obj_placed": is_obj_placed,
            "is_right_static": is_right_static,
            "is_left_static": is_left_static,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                goal_pos=self.goal_site.pose.p,
                left_tcp_to_obj_pos=self.cube.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.cube.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        right_tcp_to_obj = torch.linalg.norm(self.cube.pose.p - self.right_agent.tcp.pose.p, axis=1)
        left_tcp_to_obj = torch.linalg.norm(self.cube.pose.p - self.left_agent.tcp.pose.p, axis=1)
        obj_to_goal = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        right_reach = 1 - torch.tanh(5 * right_tcp_to_obj)
        left_support = 1 - torch.tanh(5 * left_tcp_to_obj)
        place = 1 - torch.tanh(5 * obj_to_goal)
        reward = right_reach + 0.5 * left_support + 2.0 * place
        reward[info["is_obj_placed"]] = 5.0
        reward[info["success"]] = 6.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0


def register_env() -> None:
    env_specs = [
        ("PickCubeMultiCam-v1", "robust_recovery.envs.pickcube_multicam:PickCubeMultiCamEnv"),
        ("PickCubeBoxMultiCam-v1", "robust_recovery.envs.pickcube_multicam:PickCubeBoxMultiCamEnv"),
        ("PhoneSlotMultiCam-v1", "robust_recovery.envs.pickcube_multicam:PhoneSlotMultiCamEnv"),
        ("TwoPandaPhoneSlotMultiCam-v1", "robust_recovery.envs.pickcube_multicam:TwoPandaPhoneSlotMultiCamEnv"),
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
