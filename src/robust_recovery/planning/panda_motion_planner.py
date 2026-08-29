"""Task-level Panda motion planner wrappers for data collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sapien

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


FINGER_LENGTH = 0.025
PANDA_HOME_QPOS = np.array(
    [0.0, np.pi / 8, 0.0, -5 * np.pi / 8, 0.0, 3 * np.pi / 4, np.pi / 4],
    dtype=np.float32,
)


@dataclass
class PickPlaceWaypoints:
    pre_grasp: sapien.Pose
    grasp: sapien.Pose
    lift: sapien.Pose
    pre_place: sapien.Pose
    place: sapien.Pose


@dataclass
class PlanningResult:
    success: bool
    failed_stage: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


def z_offset(height: float) -> np.ndarray:
    return np.array([0.0, 0.0, height], dtype=np.float32)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm


def quat_lerp(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    if float(np.dot(q1, q2)) < 0.0:
        q2 = -q2
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return quat_normalize((1.0 - alpha) * q1 + alpha * q2)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float32)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float32)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_inverse(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_angle_deg(q: np.ndarray) -> float:
    q = quat_normalize(q)
    return float(np.rad2deg(2.0 * np.arccos(np.clip(abs(float(q[0])), -1.0, 1.0))))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat_normalize(np.asarray(q, dtype=np.float32))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def vector_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0
    cos_angle = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def make_top_down_pose(position: np.ndarray) -> sapien.Pose:
    """Create a top-down Panda TCP pose at a world-frame position."""
    return sapien.Pose(
        p=np.asarray(position, dtype=np.float32),
        q=[0.0, 1.0, 0.0, 0.0],
    )


def build_pick_place_waypoints(
    obj_pos: np.ndarray,
    goal_pos: np.ndarray,
    pre_grasp_height: float = 0.10,
    grasp_height: float = 0.025,
    lift_height: float = 0.18,
    pre_place_height: float = 0.12,
    place_height: float = 0.035,
) -> PickPlaceWaypoints:
    """Build world-frame TCP waypoints for a simple pick-and-place task."""
    obj_pos = np.asarray(obj_pos, dtype=np.float32)
    goal_pos = np.asarray(goal_pos, dtype=np.float32)

    return PickPlaceWaypoints(
        pre_grasp=make_top_down_pose(obj_pos + z_offset(pre_grasp_height)),
        grasp=make_top_down_pose(obj_pos + z_offset(grasp_height)),
        lift=make_top_down_pose(obj_pos + z_offset(lift_height)),
        pre_place=make_top_down_pose(goal_pos + z_offset(pre_place_height)),
        place=make_top_down_pose(goal_pos + z_offset(place_height)),
    )


class PandaPickPlacePlanner:
    """High-level pick-and-place wrapper around ManiSkill's Panda planner."""

    def __init__(
        self,
        env,
        debug: bool = False,
        vis: bool = False,
        print_env_info: bool = False,
        joint_vel_limits: float = 0.55,
        joint_acc_limits: float = 0.45,
        prefer_screw: bool = True,
        grasp_diversity: bool = False,
        grasp_candidate_count: int = 4,
        refine_scale: int = 1,
        rotate_on_approach: bool = False,
        rng_seed: int | None = None,
        pre_place_height: float = 0.10,
        place_height: float | None = None,
        return_home: bool = False,
        home_qpos: np.ndarray | None = None,
    ) -> None:
        self.env = env
        self.prefer_screw = prefer_screw
        self.grasp_diversity = grasp_diversity
        self.grasp_candidate_count = max(1, grasp_candidate_count)
        self.refine_scale = max(1, refine_scale)
        self.rotate_on_approach = rotate_on_approach
        self.return_home = return_home
        self.home_qpos = np.asarray(PANDA_HOME_QPOS if home_qpos is None else home_qpos, dtype=np.float32)
        self.rng = np.random.default_rng(rng_seed)
        self.last_candidate_index = 0
        self.last_candidate_label = "default"
        self.last_closing_direction: list[float] | None = None
        self.dynamic_obstacle_agent = None
        self.dynamic_obstacle_name = "other_arm_dynamic_obstacle"
        self._active_stage: str | None = None
        self.local_cartesian_infos: list[dict[str, Any]] = []
        base_env = env.unwrapped
        self.base_env = base_env
        if place_height is None:
            env_place_height = getattr(base_env, "planner_default_place_height", None)
            if env_place_height is not None:
                place_height = float(env_place_height)
            else:
                place_height = 0.020 if hasattr(base_env, "goal_box") and not getattr(base_env, "goal_box_follow_goal", True) else 0.035
        self.pre_place_height = pre_place_height
        self.place_height = place_height
        robot_base_pose = base_env.agent.robot.pose

        self.solver = PandaArmMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=robot_base_pose,
            visualize_target_grasp_pose=vis,
            print_env_info=print_env_info,
            joint_vel_limits=joint_vel_limits,
            joint_acc_limits=joint_acc_limits,
        )

    def move(self, pose: sapien.Pose, refine_steps: int = 0) -> bool:
        """Plan and execute motion to a target TCP pose."""
        result = self.plan_pose(pose)
        if result == -1:
            return False
        self.solver.follow_path(result, refine_steps=refine_steps)
        return True

    def plan_pose(self, pose: sapien.Pose):
        """Plan a pose motion without executing it."""
        self._update_dynamic_obstacle_point_cloud()
        if self.prefer_screw:
            result = self.solver.move_to_pose_with_screw(
                pose,
                dry_run=True,
            )
            if result == -1:
                result = self.solver.move_to_pose_with_RRTConnect(
                    pose,
                    dry_run=True,
                )
        else:
            result = self.solver.move_to_pose_with_RRTConnect(
                pose,
                dry_run=True,
            )
            if result == -1:
                result = self.solver.move_to_pose_with_screw(
                    pose,
                    dry_run=True,
                )

        if result == -1:
            result = self.solver.move_to_pose_with_RRTStar(
                pose,
                dry_run=True,
            )

        return result

    def _stage_uses_dynamic_obstacle(self) -> bool:
        env = self.env.unwrapped
        if not bool(getattr(env, "planner_other_arm_obstacle_enabled", True)):
            return False
        stage = self._active_stage or ""
        skip_tokens = tuple(
            getattr(
                env,
                "planner_other_arm_obstacle_skip_stage_tokens",
                (
                    "pre_grasp",
                    "grasp",
                    "receive",
                    "close_gripper",
                    "confirm_handoff",
                    "release_handoff",
                ),
            )
        )
        return not any(token in stage for token in skip_tokens)

    def _link_world_position(self, link) -> np.ndarray | None:
        try:
            pos = np.asarray(link.pose.sp.p, dtype=np.float32)
        except Exception:
            return None
        pos = np.squeeze(pos)
        if pos.shape != (3,):
            return None
        return pos

    def _sample_obstacle_sphere(self, center: np.ndarray, radius: float) -> list[np.ndarray]:
        offsets = [
            (0.0, 0.0, 0.0),
            (radius, 0.0, 0.0),
            (-radius, 0.0, 0.0),
            (0.0, radius, 0.0),
            (0.0, -radius, 0.0),
            (0.0, 0.0, radius),
            (0.0, 0.0, -radius),
        ]
        return [center + np.asarray(offset, dtype=np.float32) for offset in offsets]

    def _other_arm_obstacle_points(self) -> np.ndarray:
        if self.dynamic_obstacle_agent is None:
            return np.empty((0, 3), dtype=np.float32)
        env = self.env.unwrapped
        radius = float(getattr(env, "planner_other_arm_obstacle_radius", 0.055))
        link_stride = int(getattr(env, "planner_other_arm_obstacle_link_stride", 1))
        link_names = set(
            str(name)
            for name in getattr(
                env,
                "planner_other_arm_obstacle_link_names",
                (
                    "panda_link7",
                    "panda_link8",
                    "panda_hand",
                    "panda_hand_tcp",
                    "panda_leftfinger",
                    "panda_rightfinger",
                    "panda_leftfinger_pad",
                    "panda_rightfinger_pad",
                ),
            )
        )
        points: list[np.ndarray] = []
        link_positions: list[np.ndarray] = []
        for index, link in enumerate(self.dynamic_obstacle_agent.robot.get_links()):
            if link_names and link.name not in link_names:
                continue
            if link_stride > 1 and index % link_stride != 0:
                continue
            pos = self._link_world_position(link)
            if pos is not None:
                link_positions.append(pos)
                points.extend(self._sample_obstacle_sphere(pos, radius))

        for start, end in zip(link_positions, link_positions[1:]):
            for alpha in (0.25, 0.5, 0.75):
                center = (1.0 - alpha) * start + alpha * end
                points.extend(self._sample_obstacle_sphere(center, radius * 0.8))

        if not points:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)

    def _update_dynamic_obstacle_point_cloud(self) -> None:
        if self.dynamic_obstacle_agent is None:
            return
        planner = self.solver.planner
        try:
            planner.remove_point_cloud(self.dynamic_obstacle_name)
        except Exception:
            pass
        if not self._stage_uses_dynamic_obstacle():
            return
        points = self._other_arm_obstacle_points()
        if points.size == 0:
            return
        resolution = float(getattr(self.env.unwrapped, "planner_other_arm_obstacle_resolution", 0.010))
        try:
            planner.update_point_cloud(points, resolution=resolution, name=self.dynamic_obstacle_name)
        except Exception:
            pass

    def _current_planner_qpos(self) -> np.ndarray:
        return self.solver.robot.get_qpos()[0].cpu().numpy().astype(np.float32)

    def _regularization_weights(self, dof: int) -> np.ndarray:
        env = self.env.unwrapped
        weights = np.asarray(
            getattr(
                env,
                "planner_joint_regularization_weights",
                [8.0, 8.0, 6.0, 4.0, 2.0, 1.0, 1.0],
            ),
            dtype=np.float32,
        )
        if weights.size < dof:
            weights = np.pad(weights, (0, dof - weights.size), constant_values=float(weights[-1]))
        return weights[:dof]

    def _trajectory_regularization_cost(
        self,
        result: dict[str, Any],
        start_qpos: np.ndarray,
    ) -> float:
        path = np.asarray(result["position"], dtype=np.float32)
        dof = min(path.shape[1], start_qpos.size)
        path = path[:, :dof]
        start = start_qpos[:dof]
        weights = self._regularization_weights(dof)

        terminal_delta = path[-1] - start
        step_delta = np.diff(np.vstack([start[None, :], path]), axis=0)
        terminal_cost = float(np.sum(weights * terminal_delta * terminal_delta))
        smooth_cost = float(np.sum(weights[None, :] * step_delta * step_delta))
        duration_cost = float(path.shape[0])

        env = self.env.unwrapped
        return (
            float(getattr(env, "planner_regularized_terminal_weight", 1.0)) * terminal_cost
            + float(getattr(env, "planner_regularized_smooth_weight", 0.25)) * smooth_cost
            + float(getattr(env, "planner_regularized_duration_weight", 0.002)) * duration_cost
        )

    def _regularized_ik_candidates(
        self,
        pose: sapien.Pose,
        start_qpos: np.ndarray,
    ) -> list[tuple[str, dict[str, Any], float]]:
        env = self.env.unwrapped
        n_init = int(getattr(env, "planner_regularized_ik_samples", 32))
        max_candidates = int(getattr(env, "planner_regularized_ik_plan_candidates", 6))
        transformed_pose = self.solver._transform_pose_for_planning(pose)
        goal_pose = self.solver._to_mplib_pose(transformed_pose)

        status, q_goals = self.solver.planner.IK(
            goal_pose,
            start_qpos,
            n_init_qpos=n_init,
            threshold=float(getattr(env, "planner_regularized_ik_threshold", 0.003)),
            return_closest=False,
            verbose=False,
        )
        if status != "Success" or q_goals is None:
            return []

        if isinstance(q_goals, np.ndarray):
            goals = [q_goals]
        else:
            goals = [np.asarray(q, dtype=np.float32) for q in q_goals]

        dof = min(start_qpos.size, len(self.solver.planner.joint_vel_limits))
        weights = self._regularization_weights(dof)
        goals = sorted(
            goals,
            key=lambda q: float(np.sum(weights * (np.asarray(q[:dof], dtype=np.float32) - start_qpos[:dof]) ** 2)),
        )

        candidates: list[tuple[str, dict[str, Any], float]] = []
        for index, goal_qpos in enumerate(goals[:max_candidates]):
            result = self.solver.planner.plan_qpos(
                [np.asarray(goal_qpos, dtype=np.float32)],
                start_qpos,
                time_step=self.solver.base_env.control_timestep,
                planning_time=float(getattr(env, "planner_regularized_planning_time", 0.8)),
                rrt_range=float(getattr(env, "planner_regularized_rrt_range", 0.08)),
            )
            if result["status"] == "Success":
                cost = self._trajectory_regularization_cost(result, start_qpos)
                candidates.append((f"ik_qpos_{index}", result, cost))
        return candidates

    def _regularized_pose_candidates(
        self,
        pose: sapien.Pose,
        start_qpos: np.ndarray,
    ) -> list[tuple[str, dict[str, Any], float]]:
        env = self.env.unwrapped
        attempts = int(getattr(env, "planner_regularized_pose_attempts", 4))
        candidates: list[tuple[str, dict[str, Any], float]] = []

        if self.prefer_screw:
            result = self.solver.move_to_pose_with_screw(pose, dry_run=True)
            if result != -1:
                candidates.append(("screw", result, self._trajectory_regularization_cost(result, start_qpos)))

        for index in range(attempts):
            result = self.solver.move_to_pose_with_RRTConnect(pose, dry_run=True)
            if result != -1:
                cost = self._trajectory_regularization_cost(result, start_qpos)
                candidates.append((f"rrt_connect_{index}", result, cost))

        if not candidates:
            result = self.solver.move_to_pose_with_RRTStar(pose, dry_run=True)
            if result != -1:
                candidates.append(("rrt_star", result, self._trajectory_regularization_cost(result, start_qpos)))
        return candidates

    def regularized_move(self, pose: sapien.Pose, refine_steps: int = 0) -> bool:
        """Plan several pose-reaching candidates and execute the lowest joint-change cost path."""
        pose = sapien.Pose(pose.p, pose.q)
        self._update_dynamic_obstacle_point_cloud()
        self.solver._update_grasp_visual(pose)
        start_qpos = self._current_planner_qpos()

        candidates = self._regularized_ik_candidates(pose, start_qpos)
        candidates.extend(self._regularized_pose_candidates(pose, start_qpos))
        if not candidates:
            return self.move(pose, refine_steps=refine_steps)

        label, result, cost = min(candidates, key=lambda item: item[2])
        self.last_regularized_motion = {
            "label": label,
            "cost": float(cost),
            "candidate_count": len(candidates),
        }
        self.solver.follow_path(result, refine_steps=refine_steps)
        return True

    def _refine(self, steps: int) -> int:
        return steps * self.refine_scale

    def move_to_qpos(self, target_qpos: np.ndarray, refine_steps: int = 0) -> bool:
        """Execute a smooth joint-position return motion for the Panda arm."""
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        arm_dof = 7
        if target_qpos.size != arm_dof:
            raise ValueError(f"Expected {arm_dof} arm joints, got {target_qpos.size}.")

        start_qpos = self.solver.robot.get_qpos()[0, :arm_dof].cpu().numpy().astype(np.float32)
        max_delta = float(np.max(np.abs(target_qpos - start_qpos)))
        n_step = max(2, int(np.ceil(max_delta / 0.015))) + refine_steps

        obs = reward = terminated = truncated = info = None
        for i in range(n_step):
            alpha = (i + 1) / n_step
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
            if self.solver.control_mode == "pd_joint_pos_vel":
                action = np.hstack([qpos, np.zeros_like(qpos), self.solver.gripper_state])
            else:
                action = np.hstack([qpos, self.solver.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.solver.elapsed_steps += 1
            if self.solver.print_env_info:
                print(f"[{self.solver.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.solver.vis:
                self.solver.base_env.render_human()

        return obs is not None

    def _actual_base_env(self):
        env = self.env.unwrapped
        return getattr(env, "actual_base_env", env)

    def _selected_agent(self):
        return self.env.unwrapped.agent

    def _current_tcp_pose(self) -> sapien.Pose:
        tcp_pose = self._selected_agent().tcp.pose.sp
        return sapien.Pose(
            np.asarray(tcp_pose.p, dtype=np.float32),
            np.asarray(tcp_pose.q, dtype=np.float32),
        )

    def _current_object_position(self) -> np.ndarray | None:
        base_env = self._actual_base_env()
        if not hasattr(base_env, "cube"):
            return None
        return np.asarray(base_env.cube.pose.sp.p, dtype=np.float32)

    def _other_tcp_distance(self) -> float | None:
        base_env = self._actual_base_env()
        if not hasattr(base_env, "left_agent") or not hasattr(base_env, "right_agent"):
            return None
        try:
            left_p = np.asarray(base_env.left_agent.tcp.pose.sp.p, dtype=np.float32)
            right_p = np.asarray(base_env.right_agent.tcp.pose.sp.p, dtype=np.float32)
        except Exception:
            return None
        return float(np.linalg.norm(left_p - right_p))

    def _step_arm_qpos(self, qpos: np.ndarray):
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)[:7]
        if self.solver.control_mode == "pd_joint_pos_vel":
            action = np.hstack([qpos, np.zeros_like(qpos), self.solver.gripper_state]).astype(np.float32)
        else:
            action = np.hstack([qpos, self.solver.gripper_state]).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.solver.elapsed_steps += 1
        if self.solver.print_env_info:
            print(f"[{self.solver.elapsed_steps:3}] Env Output: reward={reward} info={info}")
        if self.solver.vis:
            self.solver.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def _ik_closest_qpos(self, pose: sapien.Pose, start_qpos: np.ndarray) -> np.ndarray | None:
        pose = sapien.Pose(pose.p, pose.q)
        self._update_dynamic_obstacle_point_cloud()
        self.solver._update_grasp_visual(pose)
        planning_pose = self.solver._transform_pose_for_planning(pose)
        threshold = float(getattr(self.base_env, "planner_local_cartesian_ik_threshold", 0.003))
        status, qpos = self.solver.planner.IK(
            self.solver._to_mplib_pose(planning_pose),
            np.asarray(start_qpos, dtype=np.float32),
            threshold=threshold,
            return_closest=True,
        )
        if status != "Success" or qpos is None:
            return None
        return np.asarray(qpos, dtype=np.float32)

    def _run_local_screw_stage(
        self,
        stage: str,
        target_pose: sapien.Pose,
        completed_stages: list[str],
        *,
        check_object_static: bool = False,
    ) -> bool:
        target_pose = sapien.Pose(target_pose.p, target_pose.q)
        self._update_dynamic_obstacle_point_cloud()
        self.solver._update_grasp_visual(target_pose)
        planning_pose = self.solver._transform_pose_for_planning(target_pose)
        start_qpos = self.solver.robot.get_qpos().cpu().numpy()[0]
        info: dict[str, Any] = {
            "stage": stage,
            "mode": "local_screw",
            "executed_steps": 0,
            "success": False,
            "failure_reason": None,
        }
        result = self.solver.planner.plan_screw(
            self.solver._to_mplib_pose(planning_pose),
            start_qpos,
            time_step=self.solver.base_env.control_timestep,
        )
        if result["status"] != "Success":
            result = self.solver.planner.plan_screw(
                self.solver._to_mplib_pose(planning_pose),
                start_qpos,
                time_step=self.solver.base_env.control_timestep,
            )
        if result["status"] != "Success":
            info["failure_reason"] = str(result["status"])
            self.local_cartesian_infos.append(info)
            return False

        positions = np.asarray(result["position"], dtype=np.float32)
        info["planned_steps"] = int(positions.shape[0])
        start_obj = self._current_object_position()
        max_obj_motion = float(getattr(self.base_env, "planner_local_cartesian_max_obj_motion", 0.004))
        min_other_tcp_distance = float(getattr(self.base_env, "planner_local_cartesian_min_other_tcp_distance", 0.025))
        previous_stage = self._active_stage
        self._active_stage = stage
        try:
            for index, qpos in enumerate(positions):
                self._step_arm_qpos(qpos)
                info["executed_steps"] = index + 1
                other_tcp_distance = self._other_tcp_distance()
                if other_tcp_distance is not None and other_tcp_distance < min_other_tcp_distance:
                    info["failure_reason"] = "other_tcp_too_close"
                    info["other_tcp_distance"] = other_tcp_distance
                    return False
                if check_object_static and start_obj is not None:
                    obj = self._current_object_position()
                    if obj is not None:
                        obj_motion = float(np.linalg.norm(obj - start_obj))
                        if obj_motion > max_obj_motion:
                            info["failure_reason"] = "object_moved_during_approach"
                            info["object_motion"] = obj_motion
                            return False
        finally:
            self._active_stage = previous_stage
            self.local_cartesian_infos.append(info)

        info["success"] = True
        completed_stages.append(stage)
        return True

    def _run_local_cartesian_stage(
        self,
        stage: str,
        target_pose: sapien.Pose,
        completed_stages: list[str],
        *,
        check_object_static: bool = False,
    ) -> bool:
        """Execute a short local Cartesian segment with per-step IK and safety checks."""
        if not bool(getattr(self.base_env, "planner_local_cartesian_grasp_enabled", True)):
            return self._run_stage(stage, target_pose, completed_stages, 0)
        mode = str(getattr(self.base_env, "planner_local_cartesian_mode", "screw"))
        if mode != "ik":
            return self._run_local_screw_stage(
                stage,
                target_pose,
                completed_stages,
                check_object_static=check_object_static,
            )

        start_pose = self._current_tcp_pose()
        start_p = np.asarray(start_pose.p, dtype=np.float32)
        target_p = np.asarray(target_pose.p, dtype=np.float32)
        distance = float(np.linalg.norm(target_p - start_p))
        step_size = max(1e-4, float(getattr(self.base_env, "planner_local_cartesian_step_size", 0.003)))
        steps = max(2, int(np.ceil(distance / step_size)))
        max_joint_delta = float(getattr(self.base_env, "planner_local_cartesian_max_joint_delta", 0.120))
        max_obj_motion = float(getattr(self.base_env, "planner_local_cartesian_max_obj_motion", 0.004))
        min_other_tcp_distance = float(getattr(self.base_env, "planner_local_cartesian_min_other_tcp_distance", 0.025))
        start_obj = self._current_object_position()
        previous_arm_qpos = self.solver.robot.get_qpos()[0, :7].cpu().numpy().astype(np.float32)
        full_start_qpos = self.solver.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
        info: dict[str, Any] = {
            "stage": stage,
            "mode": "local_cartesian_ik",
            "distance": distance,
            "step_size": step_size,
            "planned_steps": int(steps),
            "executed_steps": 0,
            "success": False,
            "failure_reason": None,
        }

        previous_stage = self._active_stage
        self._active_stage = stage
        try:
            for index in range(steps):
                alpha = float(index + 1) / float(steps)
                pose = sapien.Pose(
                    (1.0 - alpha) * start_p + alpha * target_p,
                    quat_lerp(np.asarray(start_pose.q, dtype=np.float32), np.asarray(target_pose.q, dtype=np.float32), alpha),
                )
                qpos = self._ik_closest_qpos(pose, full_start_qpos)
                if qpos is None:
                    info["failure_reason"] = "ik_failed"
                    return False
                arm_qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)[:7]
                joint_delta = float(np.max(np.abs(arm_qpos - previous_arm_qpos)))
                if joint_delta > max_joint_delta:
                    info["failure_reason"] = "joint_delta_too_large"
                    info["joint_delta"] = joint_delta
                    return False
                self._step_arm_qpos(arm_qpos)
                info["executed_steps"] = index + 1
                previous_arm_qpos = arm_qpos
                full_start_qpos = self.solver.robot.get_qpos()[0].cpu().numpy().astype(np.float32)

                other_tcp_distance = self._other_tcp_distance()
                if other_tcp_distance is not None and other_tcp_distance < min_other_tcp_distance:
                    info["failure_reason"] = "other_tcp_too_close"
                    info["other_tcp_distance"] = other_tcp_distance
                    return False
                if check_object_static and start_obj is not None:
                    obj = self._current_object_position()
                    if obj is not None:
                        obj_motion = float(np.linalg.norm(obj - start_obj))
                        if obj_motion > max_obj_motion:
                            info["failure_reason"] = "object_moved_during_approach"
                            info["object_motion"] = obj_motion
                            return False
        finally:
            self._active_stage = previous_stage
            self.local_cartesian_infos.append(info)

        info["success"] = True
        completed_stages.append(stage)
        return True

    def _run_stage(
        self,
        stage: str,
        pose: sapien.Pose,
        completed_stages: list[str],
        refine_steps: int = 0,
        regularized: bool = False,
    ) -> bool:
        previous_stage = self._active_stage
        self._active_stage = stage
        try:
            ok = (
                self.regularized_move(pose, refine_steps=refine_steps)
                if regularized
                else self.move(pose, refine_steps=refine_steps)
            )
        finally:
            self._active_stage = previous_stage
        if ok:
            completed_stages.append(stage)
        return ok

    def _build_pickcube_waypoint_candidates(
        self,
        goal_pos: np.ndarray,
        lift_height: float = 0.12,
        pre_place_height: float | None = None,
        place_height: float | None = None,
    ) -> list[tuple[str, np.ndarray, PickPlaceWaypoints]]:
        env = self.env.unwrapped
        if pre_place_height is None:
            pre_place_height = self.pre_place_height
        if place_height is None:
            place_height = self.place_height
        if not all(hasattr(env, name) for name in ("agent", "cube")):
            return []

        if not self.rotate_on_approach:
            tcp_pose = env.agent.tcp.pose.sp
            cube_pos = np.asarray(env.cube.pose.sp.p, dtype=np.float32)
            goal_pos = np.asarray(goal_pos, dtype=np.float32)
            tcp_q = np.asarray(tcp_pose.q, dtype=np.float32)
            grasp_z_offset = float(getattr(env, "planner_grasp_z_offset", 0.0))
            grasp_pos = cube_pos + z_offset(grasp_z_offset)
            grasp_pose = sapien.Pose(grasp_pos, tcp_q)
            insert_rotation_q = getattr(env, "planner_insert_rotation_q", None)
            place_q = tcp_q
            if insert_rotation_q is not None:
                place_q = quat_mul(np.asarray(insert_rotation_q, dtype=np.float32), tcp_q)
            return [
                (
                    "no_rotate_current_tcp",
                    np.asarray(
                        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy(),
                        dtype=np.float32,
                    ),
                    PickPlaceWaypoints(
                        pre_grasp=grasp_pose * sapien.Pose([0, 0, -0.05]),
                        grasp=grasp_pose,
                        lift=sapien.Pose(grasp_pos + z_offset(lift_height), tcp_q),
                        pre_place=sapien.Pose(goal_pos + z_offset(pre_place_height), place_q),
                        place=sapien.Pose(goal_pos + z_offset(place_height), place_q),
                    ),
                )
            ]

        obb = get_actor_obb(env.cube)
        grasp_z_offset = float(getattr(env, "planner_grasp_z_offset", 0.0))
        grasp_target = np.asarray(env.cube.pose.sp.p, dtype=np.float32) + z_offset(grasp_z_offset)
        approaching = np.array([0, 0, -1])
        closing_candidates = self._closing_candidates()
        goal_pos = np.asarray(goal_pos, dtype=np.float32)

        waypoints: list[tuple[str, np.ndarray, PickPlaceWaypoints]] = []
        seen_closing: list[np.ndarray] = []

        for label, target_closing in closing_candidates:
            if self.grasp_diversity:
                closing = np.asarray(target_closing, dtype=np.float32)
            else:
                grasp_info = compute_grasp_info_by_obb(
                    obb,
                    approaching=approaching,
                    target_closing=target_closing,
                    depth=FINGER_LENGTH,
                )
                closing = np.asarray(grasp_info["closing"], dtype=np.float32)

            if any(np.dot(closing, seen) > 0.99 for seen in seen_closing):
                continue
            seen_closing.append(closing)

            grasp_pose = env.agent.build_grasp_pose(
                approaching,
                closing,
                grasp_target,
            )
            waypoints.append(
                (
                    label,
                    closing,
                    PickPlaceWaypoints(
                        pre_grasp=grasp_pose * sapien.Pose([0, 0, -0.05]),
                        grasp=grasp_pose,
                        lift=sapien.Pose(grasp_pose.p + z_offset(lift_height), grasp_pose.q),
                        pre_place=sapien.Pose(goal_pos + z_offset(pre_place_height), grasp_pose.q),
                        place=sapien.Pose(goal_pos + z_offset(place_height), grasp_pose.q),
                    ),
                )
            )

        return waypoints

    def _closing_candidates(self) -> list[tuple[str, np.ndarray]]:
        env = self.env.unwrapped
        current_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        base_candidates = [
            ("current_tcp_y", current_closing),
            ("x_pos", np.array([1.0, 0.0, 0.0])),
            ("y_pos", np.array([0.0, 1.0, 0.0])),
            ("x_neg", np.array([-1.0, 0.0, 0.0])),
            ("y_neg", np.array([0.0, -1.0, 0.0])),
        ]

        candidates: list[tuple[str, np.ndarray]] = []
        for label, candidate in base_candidates:
            candidate = np.asarray(candidate, dtype=np.float32)
            candidate[2] = 0.0
            norm = np.linalg.norm(candidate)
            if norm < 1e-6:
                continue
            candidate = candidate / norm
            if not any(np.dot(candidate, seen) > 0.99 for _, seen in candidates):
                candidates.append((label, candidate))

        if self.grasp_diversity and len(candidates) > 1:
            order = self.rng.permutation(len(candidates))
            candidates = [candidates[i] for i in order]

        return candidates[: self.grasp_candidate_count]

    def _select_waypoints(
        self,
        candidates: list[tuple[str, np.ndarray, PickPlaceWaypoints]],
    ) -> PickPlaceWaypoints | None:
        if not candidates:
            return None

        for index, (label, closing, waypoints) in enumerate(candidates):
            result = self.solver.move_to_pose_with_screw(waypoints.pre_grasp, dry_run=True)
            if result == -1:
                result = self.solver.move_to_pose_with_RRTConnect(
                    waypoints.pre_grasp,
                    dry_run=True,
                )
            if result != -1:
                self.last_candidate_index = index
                self.last_candidate_label = label
                self.last_closing_direction = closing.astype(float).tolist()
                return waypoints

        self.last_candidate_index = 0
        self.last_candidate_label = candidates[0][0]
        self.last_closing_direction = candidates[0][1].astype(float).tolist()
        return candidates[0][2]

    def _build_waypoints(
        self,
        obj_pos: np.ndarray,
        goal_pos: np.ndarray,
    ) -> PickPlaceWaypoints:
        waypoints = self._select_waypoints(
            self._build_pickcube_waypoint_candidates(goal_pos)
        )
        if waypoints is not None:
            return waypoints
        return build_pick_place_waypoints(
            obj_pos,
            goal_pos,
            pre_place_height=self.pre_place_height,
            place_height=self.place_height,
        )

    def pick_and_place(
        self,
        obj_pos: np.ndarray,
        goal_pos: np.ndarray,
        close_steps: int = 10,
        open_steps: int = 10,
    ) -> PlanningResult:
        """Execute a planned pick-and-place sequence."""
        waypoints = self._build_waypoints(obj_pos, goal_pos)
        completed: list[str] = []

        stages = [
            ("pre_grasp", waypoints.pre_grasp, self._refine(2)),
            ("grasp", waypoints.grasp, self._refine(8)),
            ("lift", waypoints.lift, self._refine(2)),
            ("pre_place", waypoints.pre_place, self._refine(2)),
            ("place", waypoints.place, self._refine(8)),
        ]

        if not self._run_stage(stages[0][0], stages[0][1], completed, stages[0][2]):
            return PlanningResult(False, "pre_grasp", completed)

        if not self._run_stage(stages[1][0], stages[1][1], completed, stages[1][2]):
            return PlanningResult(False, "grasp", completed)

        self.solver.close_gripper(t=self._refine(close_steps))
        completed.append("close_gripper")

        if not self._run_stage(stages[2][0], stages[2][1], completed, stages[2][2]):
            return PlanningResult(False, "lift", completed)

        if not self._run_stage(stages[3][0], stages[3][1], completed, stages[3][2]):
            return PlanningResult(False, "pre_place", completed)

        if not self._run_stage(stages[4][0], stages[4][1], completed, stages[4][2]):
            return PlanningResult(False, "place", completed)

        self.solver.open_gripper(t=self._refine(open_steps))
        completed.append("open_gripper")

        if self.return_home:
            if not self.move_to_qpos(self.home_qpos, refine_steps=self._refine(4)):
                return PlanningResult(False, "return_home", completed)
            completed.append("return_home")

        return PlanningResult(
            True,
            None,
            completed,
            info={
                "grasp_candidate_index": self.last_candidate_index,
                "grasp_candidate_label": self.last_candidate_label,
                "closing_direction": self.last_closing_direction,
                "return_home": self.return_home,
                "home_qpos": self.home_qpos.astype(float).tolist() if self.return_home else None,
            },
        )

    def close(self) -> None:
        self.solver.close()


class SingleAgentControlAdapter:
    """Expose one arm of a ManiSkill MultiAgent env as a single-agent env.

    The wrapped planner sees only ``agent`` for the selected arm. Each emitted
    single-arm action is expanded into a MultiAgent action dict while every
    non-selected arm holds its current joint position and last commanded
    gripper state.
    """

    def __init__(
        self,
        env,
        agent_uid: str,
        agent,
        gripper_command_by_uid: dict[str, float] | None = None,
    ) -> None:
        self.env = env
        self.agent_uid = agent_uid
        self._agent = agent
        self._gripper_command_by_uid = gripper_command_by_uid if gripper_command_by_uid is not None else {}

    @property
    def unwrapped(self):
        return self

    @property
    def actual_base_env(self):
        return self.env.unwrapped

    @property
    def agent(self):
        return self._agent

    @property
    def control_mode(self):
        return self._agent.control_mode

    def __getattr__(self, name: str):
        return getattr(self.actual_base_env, name)

    def _infer_gripper_state(self, agent) -> float:
        qpos = agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
        if qpos.size >= 9 and float(np.mean(qpos[7:9])) < 0.025:
            return -1.0
        return 1.0

    def _agent_uid(self, agent) -> str | None:
        for uid, candidate in self.actual_base_env.agent.agents_dict.items():
            if candidate is agent:
                return uid
        return None

    def _remember_gripper_command(self, uid: str, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size == 0:
            return
        self._gripper_command_by_uid[uid] = float(np.clip(action[-1], -1.0, 1.0))

    def _held_gripper_state(self, agent) -> float:
        uid = self._agent_uid(agent)
        if uid is not None and uid in self._gripper_command_by_uid:
            return float(self._gripper_command_by_uid[uid])
        return self._infer_gripper_state(agent)

    def _hold_action(self, agent) -> np.ndarray:
        qpos = agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
        arm_qpos = qpos[:7]
        gripper_state = self._held_gripper_state(agent)
        if agent.control_mode == "pd_joint_pos_vel":
            return np.hstack([arm_qpos, np.zeros_like(arm_qpos), gripper_state]).astype(np.float32)
        return np.hstack([arm_qpos, gripper_state]).astype(np.float32)

    def _compose_action(self, target_action) -> dict[str, np.ndarray]:
        action_dict: dict[str, np.ndarray] = {}
        target_action = np.asarray(target_action, dtype=np.float32)
        self._remember_gripper_command(self.agent_uid, target_action)
        for uid, agent in self.actual_base_env.agent.agents_dict.items():
            if uid == self.agent_uid:
                action_dict[uid] = target_action
            else:
                action_dict[uid] = self._hold_action(agent)
        return action_dict

    def step(self, action):
        return self.env.step(self._compose_action(action))

    def close(self) -> None:
        return None


class PhoneSlotPlanner(PandaPickPlacePlanner):
    """Phone-specific planner for horizontal pickup, upright rotation, and slot insertion."""

    def _phone_grasp_poses(
        self,
        obj_pos: np.ndarray,
    ) -> tuple[sapien.Pose, sapien.Pose, sapien.Pose, np.ndarray, np.ndarray]:
        env = self.env.unwrapped
        tcp_q = np.asarray(env.agent.tcp.pose.sp.q, dtype=np.float32)
        insert_rotation_q = np.asarray(
            getattr(env, "planner_insert_rotation_q", [0.7071068, 0.0, -0.7071068, 0.0]),
            dtype=np.float32,
        )
        insert_q = quat_mul(insert_rotation_q, tcp_q)

        obj_pos = np.asarray(obj_pos, dtype=np.float32)
        phone_half_size = np.asarray(getattr(env, "phone_half_size", (0.06, 0.02, 0.002)), dtype=np.float32)
        conveyor_top_z = float(getattr(env, "conveyor_top_z", 0.0))
        grasp_z = max(float(obj_pos[2]), conveyor_top_z + float(phone_half_size[2]) + 0.001)

        grasp_pos = obj_pos.copy()
        grasp_pos[2] = grasp_z
        pre_grasp_height = float(getattr(env, "planner_right_pre_grasp_height", 0.080))
        lift_height = float(getattr(env, "planner_right_lift_height", 0.160))
        pre_grasp = sapien.Pose(grasp_pos + z_offset(pre_grasp_height), tcp_q)
        grasp = sapien.Pose(grasp_pos, tcp_q)
        lift_flat = sapien.Pose(grasp_pos + z_offset(lift_height), tcp_q)
        return pre_grasp, grasp, lift_flat, tcp_q, insert_q

    def _current_tcp_minus_obj(self) -> np.ndarray:
        env = self.env.unwrapped
        tcp_pos = np.asarray(env.agent.tcp.pose.sp.p, dtype=np.float32)
        obj_pos = np.asarray(env.cube.pose.sp.p, dtype=np.float32)
        return tcp_pos - obj_pos

    def rotate_one_joint_upright(self, completed: list[str]) -> bool:
        env = self.env.unwrapped
        joint_index = int(getattr(env, "planner_rotate_joint_index", 5))
        joint_delta = float(getattr(env, "planner_rotate_joint_delta", -np.pi / 2))
        qpos = self.solver.robot.get_qpos()[0, :7].cpu().numpy().astype(np.float32)
        target_qpos = qpos.copy()
        target_qpos[joint_index] += joint_delta
        ok = self.move_to_qpos(target_qpos, refine_steps=self._refine(12))
        if ok:
            completed.append("rotate_single_joint")
        return ok

    def _run_rotation_stage(
        self,
        stage: str,
        rotate_pos: np.ndarray,
        start_q: np.ndarray,
        end_q: np.ndarray,
        alphas: tuple[float, ...],
        completed: list[str],
    ) -> bool:
        for alpha in alphas:
            pose = sapien.Pose(rotate_pos, quat_lerp(start_q, end_q, alpha))
            if not self.move(pose, refine_steps=self._refine(4)):
                return False
        completed.append(stage)
        return True


    def pick_and_place(
        self,
        obj_pos: np.ndarray,
        goal_pos: np.ndarray,
        close_steps: int = 24,
        open_steps: int = 16,
    ) -> PlanningResult:
        """Execute a phone-specific horizontal-pick, rotate, and vertical-insert sequence."""
        completed: list[str] = []
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        pre_grasp, grasp, lift_flat, tcp_q, insert_q = self._phone_grasp_poses(obj_pos)

        if not self._run_stage("pre_grasp", pre_grasp, completed, self._refine(int(getattr(self.base_env, "planner_pre_grasp_refine_steps", 2)))):
            return PlanningResult(False, "pre_grasp", completed)
        if not self._run_local_cartesian_stage("grasp", grasp, completed, check_object_static=True):
            return PlanningResult(False, "grasp", completed)

        self.solver.close_gripper(t=self._refine(close_steps))
        completed.append("close_gripper")

        if not self._run_local_cartesian_stage("lift_flat", lift_flat, completed):
            return PlanningResult(False, "lift_flat", completed)

        use_single_joint_rotation = getattr(self.env.unwrapped, "planner_use_single_joint_rotation", False)
        if use_single_joint_rotation:
            if not self.rotate_one_joint_upright(completed):
                return PlanningResult(False, "rotate_single_joint", completed)
            insert_q = np.asarray(self.env.unwrapped.agent.tcp.pose.sp.q, dtype=np.float32)
        else:
            rotate_pos = np.asarray(self.env.unwrapped.agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(0.04)
            target_angle = float(getattr(self.env.unwrapped, "planner_insert_angle_deg", 90.0))
            rotation_alphas = tuple(
                float(alpha)
                for alpha in getattr(
                    self.env.unwrapped,
                    "planner_rotation_alphas",
                    (1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 5.0 / 6.0, 1.0),
                )
            )
            first_stage = tuple(alpha for alpha in rotation_alphas if alpha <= 0.5)
            second_stage = tuple(alpha for alpha in rotation_alphas if alpha > 0.5)
            rotation_stages = []
            if first_stage:
                rotation_stages.append((f"rotate_upright_{int(round(target_angle * first_stage[-1]))}", first_stage))
            if second_stage:
                rotation_stages.append((f"rotate_upright_{int(round(target_angle * second_stage[-1]))}", second_stage))
            for stage, alphas in rotation_stages:
                if not self._run_rotation_stage(stage, rotate_pos, tcp_q, insert_q, alphas, completed):
                    return PlanningResult(False, stage, completed)

        tcp_minus_obj = self._current_tcp_minus_obj()
        pre_insert_height = float(getattr(self.base_env, "planner_right_pre_insert_height", getattr(self.base_env, "planner_left_pre_insert_height", 0.120)))
        pre_insert_pos = goal_pos + tcp_minus_obj + z_offset(pre_insert_height)
        insert_pos = goal_pos + tcp_minus_obj

        use_regularized_insert = getattr(self.env.unwrapped, "planner_use_regularized_insert", True)
        for stage, pose, refine in [
            ("pre_insert", sapien.Pose(pre_insert_pos, insert_q), self._refine(4)),
            ("insert", sapien.Pose(insert_pos, insert_q), self._refine(14)),
        ]:
            if not self._run_stage(stage, pose, completed, refine, regularized=use_regularized_insert):
                return PlanningResult(False, stage, completed)

        self.solver.open_gripper(t=self._refine(open_steps))
        completed.append("open_gripper")

        if self.return_home:
            if not self.move_to_qpos(self.home_qpos, refine_steps=self._refine(4)):
                return PlanningResult(False, "return_home", completed)
            completed.append("return_home")

        return PlanningResult(
            True,
            None,
            completed,
            info={
                "grasp_candidate_index": 0,
                "grasp_candidate_label": (
                    "phone_horizontal_pick_single_joint_insert"
                    if use_single_joint_rotation
                    else "phone_horizontal_pick_upright_insert_dynamic_offset"
                ),
                "closing_direction": None,
                "tcp_minus_obj_at_insert": tcp_minus_obj.astype(float).tolist(),
                "use_regularized_insert": use_regularized_insert,
                "planner_insert_angle_deg": float(getattr(self.env.unwrapped, "planner_insert_angle_deg", 90.0)),
                "planner_rotation_alphas": list(getattr(self.env.unwrapped, "planner_rotation_alphas", [])),
                "last_regularized_motion": getattr(self, "last_regularized_motion", None),
                "joint_regularization_weights": self._regularization_weights(7).astype(float).tolist(),
                "return_home": self.return_home,
                "home_qpos": self.home_qpos.astype(float).tolist() if self.return_home else None,
            },
        )

class TwoPandaPhoneSlotPlanner:
    """Staged two-arm phone-slot planner.

    The right arm performs the original phone pickup/rotation/insertion. The
    left arm moves near the grasped phone after lift and then holds position as
    a passive support while the right arm rotates and inserts.
    """

    def __init__(
        self,
        env,
        debug: bool = False,
        vis: bool = False,
        print_env_info: bool = False,
        joint_vel_limits: float = 0.55,
        joint_acc_limits: float = 0.45,
        prefer_screw: bool = True,
        grasp_diversity: bool = False,
        grasp_candidate_count: int = 4,
        refine_scale: int = 1,
        rotate_on_approach: bool = False,
        rng_seed: int | None = None,
        pre_place_height: float = 0.10,
        place_height: float | None = None,
        return_home: bool = False,
        home_qpos: np.ndarray | None = None,
    ) -> None:
        self.env = env
        self.base_env = env.unwrapped
        agent_items = list(self.base_env.agent.agents_dict.items())
        if len(agent_items) != 2:
            raise ValueError("TwoPandaPhoneSlotPlanner requires exactly two agents.")

        self.left_uid, self.left_agent = agent_items[0]
        self.right_uid, self.right_agent = agent_items[1]
        self.left_initial_qpos = self.left_agent.robot.get_qpos()[0, :7].cpu().numpy().astype(np.float32)
        self.right_initial_qpos = self.right_agent.robot.get_qpos()[0, :7].cpu().numpy().astype(np.float32)
        self.gripper_command_by_uid: dict[str, float] = {}
        self.return_home = return_home

        common_kwargs = dict(
            debug=debug,
            vis=vis,
            print_env_info=print_env_info,
            joint_vel_limits=joint_vel_limits,
            joint_acc_limits=joint_acc_limits,
            prefer_screw=prefer_screw,
            grasp_diversity=grasp_diversity,
            grasp_candidate_count=grasp_candidate_count,
            refine_scale=refine_scale,
            rotate_on_approach=rotate_on_approach,
            rng_seed=rng_seed,
            pre_place_height=pre_place_height,
            place_height=place_height,
        )
        self.right = PhoneSlotPlanner(
            SingleAgentControlAdapter(env, self.right_uid, self.right_agent, self.gripper_command_by_uid),
            return_home=False,
            home_qpos=home_qpos,
            **common_kwargs,
        )
        self.left = PhoneSlotPlanner(
            SingleAgentControlAdapter(env, self.left_uid, self.left_agent, self.gripper_command_by_uid),
            return_home=False,
            home_qpos=self.left_initial_qpos,
            **common_kwargs,
        )
        self.left.dynamic_obstacle_agent = self.right_agent
        self.right.dynamic_obstacle_agent = self.left_agent
        self.stage_snapshots: list[dict[str, Any]] = []
        self.handoff_confirm_infos: list[dict[str, Any]] = []
        self.handoff_center_infos: list[dict[str, Any]] = []
        self.left_receive_attempt_infos: list[dict[str, Any]] = []
        self.state_close_infos: list[dict[str, Any]] = []
        self.object_align_infos: list[dict[str, Any]] = []
        self.insert_readiness_infos: list[dict[str, Any]] = []
        self.last_right_object_align_info: dict[str, Any] | None = None
        self.last_left_object_align_info: dict[str, Any] | None = None
        self.last_right_insert_readiness_info: dict[str, Any] | None = None
        self.last_left_insert_readiness_info: dict[str, Any] | None = None
        self.idle_returned_home_sides: set[str] = set()

    def _right_phone_grasp_poses(
        self,
        obj_pos: np.ndarray,
    ) -> tuple[sapien.Pose, sapien.Pose, sapien.Pose, np.ndarray, np.ndarray]:
        env = self.base_env
        obj_pos = np.asarray(obj_pos, dtype=np.float32)
        phone_half_size = np.asarray(getattr(env, "phone_half_size", (0.06, 0.02, 0.002)), dtype=np.float32)
        conveyor_top_z = float(getattr(env, "conveyor_top_z", 0.0))
        grasp_z = max(float(obj_pos[2]), conveyor_top_z + float(phone_half_size[2]) + 0.001)
        grasp_pos = obj_pos.copy()
        grasp_pos[2] = grasp_z

        approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        target_closing = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        try:
            grasp_info = compute_grasp_info_by_obb(
                get_actor_obb(env.cube),
                approaching=approaching,
                target_closing=target_closing,
                depth=FINGER_LENGTH,
            )
            closing = np.asarray(grasp_info["closing"], dtype=np.float32)
            closing[2] = 0.0
            if np.linalg.norm(closing) < 1e-6:
                closing = target_closing
            else:
                closing = closing / np.linalg.norm(closing)
            grasp_pos = np.asarray(grasp_info["center"], dtype=np.float32)
            grasp_pos[2] = max(float(grasp_pos[2]), grasp_z)
        except Exception:
            closing = target_closing
        grasp_pose = self.right_agent.build_grasp_pose(approaching, closing, grasp_pos)
        tcp_q = np.asarray(grasp_pose.q, dtype=np.float32)
        insert_rotation_q = np.asarray(
            getattr(env, "planner_insert_rotation_q", [0.7071068, 0.0, -0.7071068, 0.0]),
            dtype=np.float32,
        )
        insert_q = quat_mul(insert_rotation_q, tcp_q)
        pre_grasp_height = float(getattr(env, "planner_right_pre_grasp_height", 0.080))
        lift_height = float(getattr(env, "planner_right_lift_height", 0.160))
        pre_grasp = self._phone_pre_grasp_pose(self.right_agent, grasp_pos, tcp_q, closing, pre_grasp_height)
        grasp = sapien.Pose(grasp_pos, tcp_q)
        lift_flat = sapien.Pose(grasp_pos + z_offset(lift_height), tcp_q)
        return pre_grasp, grasp, lift_flat, tcp_q, insert_q


    def _left_phone_grasp_poses(
        self,
        obj_pos: np.ndarray,
    ) -> tuple[sapien.Pose, sapien.Pose, sapien.Pose, np.ndarray, np.ndarray]:
        env = self.base_env
        obj_pos = np.asarray(obj_pos, dtype=np.float32)
        phone_half_size = np.asarray(getattr(env, "phone_half_size", (0.06, 0.02, 0.002)), dtype=np.float32)
        conveyor_top_z = float(getattr(env, "conveyor_top_z", 0.0))
        grasp_z = max(float(obj_pos[2]), conveyor_top_z + float(phone_half_size[2]) + 0.001)
        grasp_pos = obj_pos.copy()
        grasp_pos[2] = grasp_z

        approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        target_closing = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        try:
            grasp_info = compute_grasp_info_by_obb(
                get_actor_obb(env.cube),
                approaching=approaching,
                target_closing=target_closing,
                depth=FINGER_LENGTH,
            )
            closing = np.asarray(grasp_info["closing"], dtype=np.float32)
            closing[2] = 0.0
            if np.linalg.norm(closing) < 1e-6:
                closing = target_closing
            else:
                closing = closing / np.linalg.norm(closing)
            grasp_pos = np.asarray(grasp_info["center"], dtype=np.float32)
            grasp_pos[2] = max(float(grasp_pos[2]), grasp_z)
        except Exception:
            closing = target_closing
        grasp_pose = self.left_agent.build_grasp_pose(approaching, closing, grasp_pos)
        tcp_q = np.asarray(grasp_pose.q, dtype=np.float32)
        insert_rotation_q = np.asarray(
            getattr(env, "planner_insert_rotation_q", [0.7071068, 0.0, -0.7071068, 0.0]),
            dtype=np.float32,
        )
        insert_q = quat_mul(insert_rotation_q, tcp_q)
        pre_grasp_height = float(getattr(env, "planner_left_pre_grasp_height", 0.080))
        lift_height = float(getattr(env, "planner_left_lift_height", 0.160))
        pre_grasp = self._phone_pre_grasp_pose(self.left_agent, grasp_pos, tcp_q, closing, pre_grasp_height)
        grasp = sapien.Pose(grasp_pos, tcp_q)
        lift_flat = sapien.Pose(grasp_pos + z_offset(lift_height), tcp_q)
        return pre_grasp, grasp, lift_flat, tcp_q, insert_q


    def _current_obj_pos(self) -> np.ndarray:
        return np.asarray(self.base_env.cube.pose.sp.p, dtype=np.float32)

    def _phone_pre_grasp_pose(
        self,
        agent,
        grasp_pos: np.ndarray,
        tcp_q: np.ndarray,
        closing: np.ndarray,
        topdown_height: float,
    ) -> sapien.Pose:
        mode = str(getattr(self.base_env, "planner_grasp_pre_approach_mode", "topdown"))
        grasp_pos = np.asarray(grasp_pos, dtype=np.float32)
        tcp_q = np.asarray(tcp_q, dtype=np.float32)
        closing = np.asarray(closing, dtype=np.float32)
        closing[2] = 0.0
        closing_norm = float(np.linalg.norm(closing))
        if closing_norm < 1e-6:
            closing = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            closing = closing / closing_norm

        if mode != "side":
            return sapien.Pose(grasp_pos + z_offset(topdown_height), tcp_q)

        tcp_pos = np.asarray(agent.tcp.pose.sp.p, dtype=np.float32)
        side_dir = closing if float(np.dot(tcp_pos - grasp_pos, closing)) >= 0.0 else -closing
        side_distance = float(getattr(self.base_env, "planner_side_pre_grasp_distance", 0.035))
        side_z = float(getattr(self.base_env, "planner_side_pre_grasp_z_offset", 0.030))
        pre_pos = grasp_pos + side_dir * side_distance + z_offset(side_z)
        return sapien.Pose(pre_pos, tcp_q)

    def _record_stage_snapshot(self, stage: str) -> None:
        obj_pose = self.base_env.cube.pose.sp
        obj_p = np.asarray(obj_pose.p, dtype=np.float32)
        obj_q = np.asarray(obj_pose.q, dtype=np.float32)
        left_pose = self.left_agent.tcp.pose.sp
        right_pose = self.right_agent.tcp.pose.sp
        left_p = np.asarray(left_pose.p, dtype=np.float32)
        right_p = np.asarray(right_pose.p, dtype=np.float32)
        self.stage_snapshots.append(
            {
                "stage": stage,
                "obj_pose": np.concatenate([obj_p, obj_q]).astype(float).tolist(),
                "left_tcp_pose": np.concatenate([left_p, np.asarray(left_pose.q, dtype=np.float32)]).astype(float).tolist(),
                "right_tcp_pose": np.concatenate([right_p, np.asarray(right_pose.q, dtype=np.float32)]).astype(float).tolist(),
                "left_tcp_minus_obj": (left_p - obj_p).astype(float).tolist(),
                "right_tcp_minus_obj": (right_p - obj_p).astype(float).tolist(),
                "gripper_command_by_uid": {uid: float(command) for uid, command in self.gripper_command_by_uid.items()},
            }
        )

    def _torch_bool_scalar(self, value: Any) -> bool | None:
        try:
            import torch
            if isinstance(value, torch.Tensor):
                return bool(value.detach().cpu().reshape(-1)[0].item())
        except Exception:
            pass
        if isinstance(value, np.ndarray):
            return bool(value.reshape(-1)[0].item())
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return None

    def _agent_is_grasping(self, agent, max_angle: float | None = None) -> bool | None:
        if not hasattr(agent, "is_grasping"):
            return None
        try:
            result = agent.is_grasping(self.base_env.cube, max_angle=max_angle) if max_angle is not None else agent.is_grasping(self.base_env.cube)
        except TypeError:
            try:
                result = agent.is_grasping(self.base_env.cube)
            except Exception:
                return None
        except Exception:
            return None
        return self._torch_bool_scalar(result)

    def _close_gripper_until_grasp(
        self,
        side: str,
        solver: PhoneSlotPlanner,
        agent,
        completed: list[str],
        stage: str,
        max_steps: int,
    ) -> bool | None:
        """Close one simulation step at a time and stop once the phone is grasped."""
        max_steps = max(1, int(max_steps))
        min_steps = min(
            max_steps,
            max(0, solver._refine(int(getattr(self.base_env, "planner_state_close_min_steps", 4)))),
        )
        max_angle = float(getattr(self.base_env, "planner_handoff_confirm_max_angle", 85.0))
        state_triggered = bool(getattr(self.base_env, "planner_state_triggered_close_enabled", True))
        grasped: bool | None = None
        steps_taken = 0

        if not state_triggered:
            solver.solver.close_gripper(t=max_steps)
            steps_taken = max_steps
            grasped = self._agent_is_grasping(agent, max_angle=max_angle)
        else:
            for step in range(max_steps):
                solver.solver.close_gripper(t=1)
                steps_taken = step + 1
                if steps_taken < min_steps:
                    continue
                grasped = self._agent_is_grasping(agent, max_angle=max_angle)
                if grasped is True:
                    break

        completed.append(stage)
        info = {
            "stage": stage,
            "side": side,
            "state_triggered": bool(state_triggered),
            "steps_taken": int(steps_taken),
            "max_steps": int(max_steps),
            "min_steps": int(min_steps),
            "is_grasping": grasped,
            "early_exit": bool(grasped is True and steps_taken < max_steps),
        }
        self.state_close_infos.append(info)
        self._record_stage_snapshot(stage)
        return grasped

    def _confirm_handoff_grasp(
        self,
        receiver_name: str,
        receiver_agent,
        completed: list[str],
        stage: str,
    ) -> bool:
        obj_pos = self._current_obj_pos()
        tcp_pos = np.asarray(receiver_agent.tcp.pose.sp.p, dtype=np.float32)
        tcp_to_obj = obj_pos - tcp_pos
        distance = float(np.linalg.norm(tcp_to_obj))
        max_distance = float(getattr(self.base_env, "planner_handoff_confirm_max_tcp_obj_dist", 0.095))
        max_angle = float(getattr(self.base_env, "planner_handoff_confirm_max_angle", 85.0))
        grasped = self._agent_is_grasping(receiver_agent, max_angle=max_angle)
        # If the direct contact check is unavailable, distance is the fallback.
        passed = (grasped is True) or (grasped is None and distance <= max_distance)
        info = {
            "stage": stage,
            "receiver": receiver_name,
            "is_grasping": grasped,
            "tcp_to_obj_distance": distance,
            "max_tcp_obj_distance": max_distance,
            "tcp_to_obj": tcp_to_obj.astype(float).tolist(),
            "obj_pos": obj_pos.astype(float).tolist(),
            "tcp_pos": tcp_pos.astype(float).tolist(),
            "passed": bool(passed),
        }
        self.handoff_confirm_infos.append(info)
        completed.append(stage)
        self._record_stage_snapshot(stage)
        return passed

    def _handoff_failure_info(self) -> dict[str, Any]:
        return {
            "handoff_confirm_infos": self.handoff_confirm_infos,
            "planner_stage_snapshots": self.stage_snapshots,
            "planner_left_receive_info": getattr(self, "last_left_receive_info", None),
            "planner_left_receive_attempt_infos": getattr(self, "left_receive_attempt_infos", []),
            "planner_right_receive_candidate": getattr(self, "last_right_receive_candidate", None),
            "planner_right_receive_attempt_infos": getattr(self, "right_receive_attempt_infos", []),
            "planner_right_receive_closed_loop_infos": getattr(self, "right_receive_closed_loop_infos", []),
            "planner_handoff_center_infos": getattr(self, "handoff_center_infos", []),
            "planner_state_close_infos": getattr(self, "state_close_infos", []),
        }

    def _target_insert_obj_q(self) -> np.ndarray:
        return quat_normalize(
            np.asarray(
                getattr(self.base_env, "planner_insert_rotation_q", [0.7071068, 0.0, -0.7071068, 0.0]),
                dtype=np.float32,
            )
        )

    def _object_alignment_info(self, side: str) -> dict[str, Any]:
        current_obj_q = quat_normalize(np.asarray(self.base_env.cube.pose.sp.q, dtype=np.float32))
        target_obj_q = self._target_insert_obj_q()
        if float(np.dot(target_obj_q, current_obj_q)) < 0.0:
            target_obj_q = -target_obj_q
        correction_q = quat_mul(target_obj_q, quat_inverse(current_obj_q))
        angle = quat_angle_deg(correction_q)
        tolerance = float(getattr(self.base_env, "planner_insert_orientation_tolerance_deg", 5.0))
        info = {
            "side": side,
            "current_obj_q": current_obj_q.astype(float).tolist(),
            "target_obj_q": target_obj_q.astype(float).tolist(),
            "correction_q": correction_q.astype(float).tolist(),
            "correction_angle_deg": float(angle),
            "tolerance_deg": tolerance,
            "aligned": bool(angle <= tolerance),
        }
        return info

    def _record_insert_readiness_info(self, side: str, info: dict[str, Any]) -> None:
        self.insert_readiness_infos.append(info)
        if side == "right":
            self.last_right_insert_readiness_info = info
        elif side == "left":
            self.last_left_insert_readiness_info = info

    def _insert_readiness_info(self, side: str, goal_pos: np.ndarray) -> dict[str, Any]:
        obj_pose = self.base_env.cube.pose.sp
        obj_pos = np.asarray(obj_pose.p, dtype=np.float32)
        current_obj_q = quat_normalize(np.asarray(obj_pose.q, dtype=np.float32))
        target_obj_q = self._target_insert_obj_q()
        if float(np.dot(target_obj_q, current_obj_q)) < 0.0:
            target_obj_q = -target_obj_q

        current_axes = quat_to_matrix(current_obj_q)
        target_axes = quat_to_matrix(target_obj_q)
        correction_q = quat_mul(target_obj_q, quat_inverse(current_obj_q))
        orientation_angle = quat_angle_deg(correction_q)

        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        lateral_error = obj_pos[:2] - goal_pos[:2]
        lateral_distance = float(np.linalg.norm(lateral_error))
        orientation_tol = float(getattr(self.base_env, "planner_insert_orientation_tolerance_deg", 5.0))
        vertical_tol = float(getattr(self.base_env, "planner_insert_vertical_tolerance_deg", orientation_tol))
        slot_axis_tol = float(getattr(self.base_env, "planner_insert_slot_axis_tolerance_deg", orientation_tol))
        lateral_tol = float(getattr(self.base_env, "planner_insert_slot_lateral_tolerance", 0.006))

        long_axis_angle = vector_angle_deg(current_axes[:, 0], target_axes[:, 0])
        width_axis_angle = vector_angle_deg(current_axes[:, 1], target_axes[:, 1])
        thickness_axis_angle = vector_angle_deg(current_axes[:, 2], target_axes[:, 2])
        vertical_aligned = long_axis_angle <= vertical_tol
        slot_axis_aligned = width_axis_angle <= slot_axis_tol and thickness_axis_angle <= slot_axis_tol
        position_aligned = lateral_distance <= lateral_tol
        orientation_aligned = orientation_angle <= orientation_tol
        ready = vertical_aligned and slot_axis_aligned and position_aligned and orientation_aligned

        return {
            "side": side,
            "obj_pos": obj_pos.astype(float).tolist(),
            "goal_pos": goal_pos.astype(float).tolist(),
            "lateral_error_xy": lateral_error.astype(float).tolist(),
            "lateral_distance_xy": lateral_distance,
            "lateral_tolerance": lateral_tol,
            "current_obj_q": current_obj_q.astype(float).tolist(),
            "target_obj_q": target_obj_q.astype(float).tolist(),
            "correction_q": correction_q.astype(float).tolist(),
            "orientation_error_deg": float(orientation_angle),
            "orientation_tolerance_deg": orientation_tol,
            "phone_long_axis_to_vertical_deg": float(long_axis_angle),
            "vertical_tolerance_deg": vertical_tol,
            "phone_width_axis_to_slot_width_deg": float(width_axis_angle),
            "phone_thickness_axis_to_slot_narrow_deg": float(thickness_axis_angle),
            "slot_axis_tolerance_deg": slot_axis_tol,
            "vertical_aligned": bool(vertical_aligned),
            "slot_axis_aligned": bool(slot_axis_aligned),
            "position_aligned": bool(position_aligned),
            "orientation_aligned": bool(orientation_aligned),
            "ready": bool(ready),
        }

    def _ensure_object_ready_for_insert(
        self,
        side: str,
        solver: PhoneSlotPlanner,
        agent,
        completed: list[str],
        goal_pos: np.ndarray,
        insert_q: np.ndarray,
    ) -> tuple[bool, np.ndarray]:
        if not bool(getattr(self.base_env, "planner_check_insert_readiness", True)):
            return True, insert_q

        attempts = int(getattr(self.base_env, "planner_insert_readiness_correction_attempts", 2))
        lift_offset = float(getattr(self.base_env, "planner_insert_readiness_correction_z_offset", 0.010))
        min_height = float(getattr(self.base_env, "planner_insert_readiness_min_height", 0.040))
        use_regularized = bool(getattr(self.base_env, "planner_use_regularized_insert", True))
        goal_pos = np.asarray(goal_pos, dtype=np.float32)

        for attempt in range(max(0, attempts) + 1):
            info = self._insert_readiness_info(side, goal_pos)
            info["attempt"] = attempt
            self._record_insert_readiness_info(side, info)
            if info["ready"]:
                completed.append(f"{side}_insert_readiness_checked")
                self._record_stage_snapshot(completed[-1])
                return True, np.asarray(agent.tcp.pose.sp.q, dtype=np.float32)
            if attempt >= max(0, attempts):
                break

            if not bool(info["orientation_aligned"] and info["vertical_aligned"] and info["slot_axis_aligned"]):
                insert_q = self._object_pose_aligned_tcp_q(side, agent)
            else:
                insert_q = np.asarray(agent.tcp.pose.sp.q, dtype=np.float32)

            obj_pos = self._current_obj_pos()
            tcp_minus_obj = np.asarray(agent.tcp.pose.sp.p, dtype=np.float32) - obj_pos
            target_obj_pos = goal_pos.copy()
            target_obj_pos[2] = max(float(obj_pos[2]) + lift_offset, float(goal_pos[2]) + min_height)
            target_tcp_pos = target_obj_pos + tcp_minus_obj
            stage = f"{side}_correct_insert_readiness" if attempt == 0 else f"{side}_correct_insert_readiness_retry_{attempt}"
            if not solver._run_stage(
                stage,
                sapien.Pose(target_tcp_pos, insert_q),
                completed,
                solver._refine(6),
                regularized=use_regularized,
            ):
                return False, insert_q
            self._record_stage_snapshot(stage)

        return False, insert_q

    def _object_pose_aligned_tcp_q(self, side: str, agent) -> np.ndarray:
        info = self._object_alignment_info(side)
        correction_q = np.asarray(info["correction_q"], dtype=np.float32)
        correction_angle = float(info["correction_angle_deg"])
        max_correction = float(
            getattr(
                self.base_env,
                f"planner_{side}_object_align_max_angle_deg",
                getattr(self.base_env, "planner_right_object_align_max_angle_deg", 20.0),
            )
        )
        applied_fraction = 1.0
        if correction_angle > max_correction > 1e-6:
            applied_fraction = max_correction / correction_angle
            correction_q = quat_lerp(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), correction_q, applied_fraction)
        current_tcp_q = np.asarray(agent.tcp.pose.sp.q, dtype=np.float32)
        aligned_tcp_q = quat_mul(correction_q, current_tcp_q)
        info["applied_correction_q"] = correction_q.astype(float).tolist()
        info["applied_fraction"] = float(applied_fraction)
        info["max_correction_deg"] = float(max_correction)
        self.object_align_infos.append(info)
        if side == "right":
            self.last_right_object_align_info = info
        elif side == "left":
            self.last_left_object_align_info = info
        return aligned_tcp_q

    def _align_object_pose_before_insert(
        self,
        side: str,
        solver: PhoneSlotPlanner,
        agent,
        completed: list[str],
        insert_q: np.ndarray,
    ) -> tuple[bool, np.ndarray]:
        if not bool(getattr(self.base_env, f"planner_{side}_align_object_pose_before_insert", getattr(self.base_env, "planner_right_align_object_pose_before_insert", True))):
            return True, insert_q
        attempts = int(getattr(self.base_env, "planner_insert_orientation_align_attempts", 2))
        align_z_offset = float(getattr(self.base_env, f"planner_{side}_object_align_z_offset", getattr(self.base_env, "planner_right_object_align_z_offset", 0.010)))
        for attempt in range(max(1, attempts)):
            info = self._object_alignment_info(side)
            if info["aligned"]:
                self.object_align_infos.append(info)
                if side == "right":
                    self.last_right_object_align_info = info
                else:
                    self.last_left_object_align_info = info
                completed.append(f"{side}_align_object_pose_checked")
                self._record_stage_snapshot(completed[-1])
                return True, np.asarray(agent.tcp.pose.sp.q, dtype=np.float32)
            insert_q = self._object_pose_aligned_tcp_q(side, agent)
            align_pos = np.asarray(agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(align_z_offset)
            stage = f"{side}_align_object_pose" if attempt == 0 else f"{side}_align_object_pose_retry_{attempt}"
            if not solver._run_stage(stage, sapien.Pose(align_pos, insert_q), completed, solver._refine(5)):
                return False, insert_q
            self._record_stage_snapshot(stage)
        return True, insert_q

    def _left_support_poses(self) -> tuple[sapien.Pose, sapien.Pose]:
        env = self.base_env
        obj_pos = self._current_obj_pos()
        support_y_offset = float(getattr(env, "planner_left_support_y_offset", -0.120))
        support_z_offset = float(getattr(env, "planner_left_support_z_offset", 0.080))
        pre_support_extra_y = float(getattr(env, "planner_left_pre_support_extra_y", -0.040))
        pre_support_extra_z = float(getattr(env, "planner_left_pre_support_extra_z", 0.060))
        tcp_q = np.asarray(self.left_agent.tcp.pose.sp.q, dtype=np.float32)
        support_pos = obj_pos + np.array([0.0, support_y_offset, support_z_offset], dtype=np.float32)
        pre_support_pos = support_pos + np.array([0.0, pre_support_extra_y, pre_support_extra_z], dtype=np.float32)
        return sapien.Pose(pre_support_pos, tcp_q), sapien.Pose(support_pos, tcp_q)

    def _run_right_rotation_and_insert(
        self,
        completed: list[str],
        goal_pos: np.ndarray,
        tcp_q: np.ndarray,
        insert_q: np.ndarray,
    ) -> PlanningResult | None:
        use_single_joint_rotation = getattr(self.base_env, "planner_use_single_joint_rotation", False)
        if use_single_joint_rotation:
            if not self.right.rotate_one_joint_upright(completed):
                return PlanningResult(False, "rotate_single_joint", completed)
            insert_q = np.asarray(self.right_agent.tcp.pose.sp.q, dtype=np.float32)
        else:
            rotate_z_offset = float(getattr(self.base_env, "planner_right_flip_z_offset", 0.040))
            rotate_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(rotate_z_offset)
            target_angle = float(getattr(self.base_env, "planner_insert_angle_deg", 90.0))
            rotation_alphas = tuple(
                float(alpha)
                for alpha in getattr(
                    self.base_env,
                    "planner_rotation_alphas",
                    (0.5, 1.0),
                )
            )
            first_stage = tuple(alpha for alpha in rotation_alphas if alpha <= 0.5)
            second_stage = tuple(alpha for alpha in rotation_alphas if alpha > 0.5)
            rotation_stages = []
            if first_stage:
                rotation_stages.append((f"rotate_upright_{int(round(target_angle * first_stage[-1]))}", first_stage))
            if second_stage:
                rotation_stages.append((f"rotate_upright_{int(round(target_angle * second_stage[-1]))}", second_stage))
            for stage, alphas in rotation_stages:
                if not self.right._run_rotation_stage(stage, rotate_pos, tcp_q, insert_q, alphas, completed):
                    return PlanningResult(False, stage, completed)

        tcp_minus_obj = self.right._current_tcp_minus_obj()
        pre_insert_height = float(getattr(self.base_env, "planner_right_pre_insert_height", 0.120))
        pre_insert_pos = goal_pos + tcp_minus_obj + z_offset(pre_insert_height)
        insert_pos = goal_pos + tcp_minus_obj
        use_regularized_insert = getattr(self.base_env, "planner_use_regularized_insert", True)
        for stage, pose, refine in [
            ("right_pre_insert", sapien.Pose(pre_insert_pos, insert_q), self.right._refine(4)),
            ("right_insert", sapien.Pose(insert_pos, insert_q), self.right._refine(14)),
        ]:
            if not self.right._run_stage(stage, pose, completed, refine, regularized=use_regularized_insert):
                return PlanningResult(False, stage, completed)
            self._record_stage_snapshot(stage)
        return None


    def _slot_id_for_goal(self, goal_pos: np.ndarray) -> int | None:
        slot_id = getattr(self.base_env, "planner_slot_id", None)
        if slot_id is not None:
            return int(slot_id)
        slot_count = int(getattr(self.base_env, "slot_count", 1))
        if slot_count <= 1:
            return None
        slot_pitch = float(getattr(self.base_env, "slot_pitch", 0.0))
        if abs(slot_pitch) < 1e-8:
            return None
        slot_center = getattr(self.base_env, "slot_center", (0.0, 0.0))
        slot_index = round((float(goal_pos[1]) - float(slot_center[1])) / slot_pitch + (slot_count - 1) / 2.0)
        return int(np.clip(slot_index, 0, slot_count - 1))

    def _select_insert_arm_for_slot(self, goal_pos: np.ndarray) -> str:
        mode = str(getattr(self.base_env, "planner_insert_arm_mode", "auto_by_slot"))
        if mode in {"left", "right"}:
            return mode
        if mode != "auto_by_slot":
            raise ValueError(f"Unsupported planner_insert_arm_mode: {mode}")

        slot_id = self._slot_id_for_goal(goal_pos)
        slot_count = int(getattr(self.base_env, "slot_count", 3))
        if slot_id is None:
            return str(getattr(self.base_env, "planner_center_slot_insert_arm", "left"))
        if slot_id <= 0:
            return "left"
        if slot_id >= slot_count - 1:
            return "right"
        center_arm = str(getattr(self.base_env, "planner_center_slot_insert_arm", "left"))
        return center_arm if center_arm in {"left", "right"} else "left"

    def _right_open_retract_and_return(
        self,
        completed: list[str],
        open_steps: int,
    ) -> PlanningResult | None:
        self.right.solver.open_gripper(t=self.right._refine(open_steps))
        completed.append("right_open_gripper")
        self._record_stage_snapshot("right_open_gripper")

        post_release_lift = float(getattr(self.base_env, "planner_right_post_release_lift_height", 0.0))
        if post_release_lift > 0.0:
            right_retract_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(post_release_lift)
            right_retract_q = np.asarray(self.right_agent.tcp.pose.sp.q, dtype=np.float32)
            if not self.right._run_stage(
                "right_retract_after_release",
                sapien.Pose(right_retract_pos, right_retract_q),
                completed,
                self.right._refine(4),
            ):
                return PlanningResult(False, "right_retract_after_release", completed)

        if self.return_home:
            idle_returned = "left" in getattr(self, "idle_returned_home_sides", set())
            if not idle_returned and not self.left.move_to_qpos(self.left_initial_qpos, refine_steps=self.left._refine(4)):
                return PlanningResult(False, "left_return_home", completed)
            if not idle_returned:
                completed.append("left_return_home")
            if not self.right.move_to_qpos(self.right_initial_qpos, refine_steps=self.right._refine(4)):
                return PlanningResult(False, "right_return_home", completed)
            completed.append("right_return_home")
        return None

    def _run_post_handoff_retract(
        self,
        side: str,
        solver: PhoneSlotPlanner,
        agent,
        completed: list[str],
        x: float,
        y: float,
        z: float,
    ) -> PlanningResult | None:
        tcp_q = np.asarray(agent.tcp.pose.sp.q, dtype=np.float32)
        start_pos = np.asarray(agent.tcp.pose.sp.p, dtype=np.float32)
        refine_steps = int(getattr(self.base_env, "planner_post_handoff_retract_refine_steps", 4))

        if abs(float(z)) > 1e-8:
            vertical_pos = start_pos + np.array([0.0, 0.0, float(z)], dtype=np.float32)
            vertical_stage = f"{side}_vertical_clearance_after_handoff"
            if not solver._run_stage(
                vertical_stage,
                sapien.Pose(vertical_pos, tcp_q),
                completed,
                solver._refine(refine_steps),
            ):
                return PlanningResult(False, vertical_stage, completed)
            self._record_stage_snapshot(vertical_stage)
        else:
            vertical_pos = start_pos

        lateral_delta = np.array([float(x), float(y), 0.0], dtype=np.float32)
        if np.linalg.norm(lateral_delta) > 1e-8:
            retract_stage = f"{side}_retract_after_handoff"
            if not solver._run_stage(
                retract_stage,
                sapien.Pose(vertical_pos + lateral_delta, tcp_q),
                completed,
                solver._refine(refine_steps),
            ):
                return PlanningResult(False, retract_stage, completed)
            self._record_stage_snapshot(retract_stage)
        return None

    def _local_cartesian_info(self) -> dict[str, Any]:
        return {
            "planner_local_cartesian_infos": self.left.local_cartesian_infos + self.right.local_cartesian_infos
            if hasattr(self, "left") and hasattr(self, "right")
            else getattr(self, "local_cartesian_infos", [])
        }

    def _gripper_command_for_agent(self, uid: str, agent) -> float:
        if uid in self.gripper_command_by_uid:
            return float(self.gripper_command_by_uid[uid])
        qpos = agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
        if qpos.size >= 9 and float(np.mean(qpos[7:9])) < 0.025:
            return -1.0
        return 1.0

    def _path_action_for_agent(
        self,
        agent,
        result: dict[str, Any],
        step_index: int,
        gripper_command: float,
    ) -> np.ndarray:
        positions = np.asarray(result["position"], dtype=np.float32)
        index = min(step_index, positions.shape[0] - 1)
        qpos = positions[index]
        if agent.control_mode == "pd_joint_pos_vel":
            velocities = np.asarray(result.get("velocity", np.zeros_like(positions)), dtype=np.float32)
            qvel = velocities[min(step_index, velocities.shape[0] - 1)] if velocities.size else np.zeros_like(qpos)
            return np.hstack([qpos, qvel, gripper_command]).astype(np.float32)
        return np.hstack([qpos, gripper_command]).astype(np.float32)

    def _qpos_action_for_agent(
        self,
        agent,
        qpos: np.ndarray,
        gripper_command: float,
    ) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)[:7]
        if agent.control_mode == "pd_joint_pos_vel":
            return np.hstack([qpos, np.zeros_like(qpos), gripper_command]).astype(np.float32)
        return np.hstack([qpos, gripper_command]).astype(np.float32)

    def _joint_return_positions(
        self,
        agent,
        target_qpos: np.ndarray,
        refine_steps: int,
        min_steps: int,
    ) -> np.ndarray:
        target_qpos = np.asarray(target_qpos, dtype=np.float32).reshape(-1)[:7]
        start_qpos = agent.robot.get_qpos()[0, :7].cpu().numpy().astype(np.float32)
        max_delta = float(np.max(np.abs(target_qpos - start_qpos)))
        n_step = max(int(min_steps), max(2, int(np.ceil(max_delta / 0.015))) + int(refine_steps))
        positions = []
        for i in range(n_step):
            alpha = (i + 1) / n_step
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            positions.append((1.0 - alpha) * start_qpos + alpha * target_qpos)
        return np.asarray(positions, dtype=np.float32)

    def _run_parallel_insert_lift_and_idle_home(
        self,
        *,
        release_side: str,
        release_solver: PhoneSlotPlanner,
        release_agent,
        insert_lift_stage: str,
        insert_solver: PhoneSlotPlanner,
        insert_agent,
        insert_lift_pose: sapien.Pose,
        insert_lift_refine: int,
        completed: list[str],
    ) -> bool:
        insert_result = insert_solver.plan_pose(insert_lift_pose)
        if insert_result == -1:
            return False

        release_uid = self.left_uid if release_side == "left" else self.right_uid
        insert_uid = self.right_uid if release_side == "left" else self.left_uid
        release_home_qpos = self.left_initial_qpos if release_side == "left" else self.right_initial_qpos

        insert_positions = np.asarray(insert_result["position"], dtype=np.float32)
        if insert_lift_refine > 0 and insert_positions.size:
            final_insert = insert_positions[-1:]
            insert_result = dict(insert_result)
            insert_result["position"] = np.concatenate(
                [insert_positions, np.repeat(final_insert, int(insert_lift_refine), axis=0)],
                axis=0,
            )
            if "velocity" in insert_result:
                velocities = np.asarray(insert_result["velocity"], dtype=np.float32)
                insert_result["velocity"] = np.concatenate(
                    [velocities, np.zeros((int(insert_lift_refine), velocities.shape[1]), dtype=np.float32)],
                    axis=0,
                )
            insert_positions = np.asarray(insert_result["position"], dtype=np.float32)

        home_positions = self._joint_return_positions(
            release_agent,
            release_home_qpos,
            release_solver._refine(4),
            min_steps=int(insert_positions.shape[0]),
        )
        n_step = max(int(insert_positions.shape[0]), int(home_positions.shape[0]))
        insert_gripper = self._gripper_command_for_agent(insert_uid, insert_agent)
        release_gripper = self._gripper_command_for_agent(release_uid, release_agent)
        insert_stage_previous = insert_solver._active_stage
        release_stage_previous = release_solver._active_stage
        insert_solver._active_stage = insert_lift_stage
        release_stage = f"{release_side}_return_home_during_{insert_lift_stage}"
        release_solver._active_stage = release_stage
        try:
            for step_index in range(n_step):
                action_dict = {
                    release_uid: self._qpos_action_for_agent(
                        release_agent,
                        home_positions[min(step_index, home_positions.shape[0] - 1)],
                        release_gripper,
                    ),
                    insert_uid: self._path_action_for_agent(
                        insert_agent,
                        insert_result,
                        step_index,
                        insert_gripper,
                    ),
                }
                obs, reward, terminated, truncated, info = self.env.step(action_dict)
                insert_solver.solver.elapsed_steps += 1
                release_solver.solver.elapsed_steps += 1
                if insert_solver.solver.print_env_info:
                    print(f"[{insert_solver.solver.elapsed_steps:3}] Env Output: reward={reward} info={info}")
                if insert_solver.solver.vis:
                    insert_solver.solver.base_env.render_human()
        finally:
            insert_solver._active_stage = insert_stage_previous
            release_solver._active_stage = release_stage_previous

        completed.append(release_stage)
        self._record_stage_snapshot(release_stage)
        completed.append(insert_lift_stage)
        self._record_stage_snapshot(insert_lift_stage)
        return True

    def _run_post_handoff_retract_with_insert_lift(
        self,
        release_side: str,
        completed: list[str],
        x: float,
        y: float,
        z: float,
        insert_lift_stage: str,
        insert_solver: PhoneSlotPlanner,
        insert_agent,
        insert_lift_pose: sapien.Pose,
        insert_lift_refine: int,
        goal_pos: np.ndarray | None = None,
    ) -> PlanningResult | None:
        release_solver = self.left if release_side == "left" else self.right
        release_agent = self.left_agent if release_side == "left" else self.right_agent
        refine_steps = int(getattr(self.base_env, "planner_post_handoff_retract_refine_steps", 4))
        release_q = np.asarray(release_agent.tcp.pose.sp.q, dtype=np.float32)
        release_start = np.asarray(release_agent.tcp.pose.sp.p, dtype=np.float32)
        if str(getattr(self.base_env, "planner_release_retract_mode", "configured")) == "away_from_slot":
            slot_center = np.asarray(getattr(self.base_env, "slot_center", (0.06, 0.0)), dtype=np.float32)
            if slot_center.size != 2:
                slot_center = np.asarray((0.06, 0.0), dtype=np.float32)
            away_xy = release_start[:2] - slot_center
            norm = float(np.linalg.norm(away_xy))
            if norm < 1e-6:
                away_xy = np.array([-1.0, 0.0], dtype=np.float32)
            else:
                away_xy = away_xy / norm
            distance = float(getattr(self.base_env, "planner_release_retract_away_from_slot_distance", 0.140))
            x = float(away_xy[0] * distance)
            y = float(away_xy[1] * distance)
            z = float(getattr(self.base_env, "planner_release_retract_away_from_slot_z", 0.0))
        vertical_pos = release_start + np.array([0.0, 0.0, float(z)], dtype=np.float32)
        vertical_stage = f"{release_side}_vertical_clearance_after_handoff"

        if abs(float(z)) > 1e-8:
            if not release_solver._run_stage(
                vertical_stage,
                sapien.Pose(vertical_pos, release_q),
                completed,
                release_solver._refine(refine_steps),
            ):
                return PlanningResult(False, vertical_stage, completed)
            self._record_stage_snapshot(vertical_stage)
        else:
            vertical_pos = release_start

        lateral_delta = np.array([float(x), float(y), 0.0], dtype=np.float32)
        if np.linalg.norm(lateral_delta) > 1e-8:
            retract_stage = f"{release_side}_retract_after_handoff"
            current_q = np.asarray(release_agent.tcp.pose.sp.q, dtype=np.float32)
            if not release_solver._run_stage(
                retract_stage,
                sapien.Pose(vertical_pos + lateral_delta, current_q),
                completed,
                release_solver._refine(refine_steps),
            ):
                return PlanningResult(False, retract_stage, completed)
            self._record_stage_snapshot(retract_stage)

        if not insert_solver._run_stage(insert_lift_stage, insert_lift_pose, completed, insert_lift_refine):
            return PlanningResult(False, insert_lift_stage, completed)
        self._record_stage_snapshot(insert_lift_stage)
        return None


    def _world_y_rotation_quat(self, angle_deg: float) -> np.ndarray:
        half = -np.deg2rad(float(angle_deg)) * 0.5
        return np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float32)

    def _handoff_center_target_pos(self) -> np.ndarray:
        center_xy = np.asarray(
            getattr(self.base_env, "planner_handoff_center_xy", (-0.08, 0.0)),
            dtype=np.float32,
        )
        if center_xy.size != 2:
            center_xy = np.asarray((-0.08, 0.0), dtype=np.float32)
        obj_pos = self._current_obj_pos()
        target = obj_pos.copy()
        target[0] = float(center_xy[0])
        target[1] = float(center_xy[1])
        center_z = getattr(self.base_env, "planner_handoff_center_z", None)
        if center_z is not None:
            target[2] = float(center_z)
        return target

    def _move_held_object_to_handoff_center(
        self,
        arm_name: str,
        planner: PhoneSlotPlanner,
        agent,
        completed: list[str],
    ) -> PlanningResult | None:
        if not bool(getattr(self.base_env, "planner_handoff_center_enabled", True)):
            return None

        attempts = int(getattr(self.base_env, "planner_handoff_center_closed_loop_attempts", 2))
        activation_xy_threshold = float(getattr(self.base_env, "planner_handoff_center_activation_xy_threshold", 0.010))
        xy_tolerance = float(getattr(self.base_env, "planner_handoff_center_xy_tolerance", 0.004))
        z_tolerance = float(getattr(self.base_env, "planner_handoff_center_z_tolerance", 0.006))
        refine_steps = int(getattr(self.base_env, "planner_handoff_center_refine_steps", 4))
        infos: list[dict[str, Any]] = []

        for attempt in range(max(1, attempts)):
            target_obj_pos = self._handoff_center_target_pos()
            obj_pos = self._current_obj_pos()
            tcp_pose = agent.tcp.pose.sp
            tcp_pos = np.asarray(tcp_pose.p, dtype=np.float32)
            tcp_q = np.asarray(tcp_pose.q, dtype=np.float32)
            tcp_minus_obj = tcp_pos - obj_pos
            target_tcp_pos = target_obj_pos + tcp_minus_obj
            error = target_obj_pos - obj_pos
            xy_error = float(np.linalg.norm(error[:2]))
            z_error = float(abs(error[2]))
            info = {
                "arm": arm_name,
                "attempt": float(attempt),
                "target_obj_pos": target_obj_pos.astype(float).tolist(),
                "obj_pos": obj_pos.astype(float).tolist(),
                "error": error.astype(float).tolist(),
                "xy_error": xy_error,
                "z_error": z_error,
                "activation_xy_threshold": activation_xy_threshold,
                "xy_tolerance": xy_tolerance,
                "z_tolerance": z_tolerance,
            }
            infos.append(info)
            if attempt == 0 and xy_error <= activation_xy_threshold and z_error <= z_tolerance:
                info["skipped_near_center"] = True
                break
            if xy_error <= xy_tolerance and z_error <= z_tolerance:
                break

            stage = f"{arm_name}_move_handoff_center" if attempt == 0 else f"{arm_name}_move_handoff_center_closed_loop_{attempt}"
            if not planner._run_stage(
                stage,
                sapien.Pose(target_tcp_pos, tcp_q),
                completed,
                planner._refine(refine_steps),
            ):
                self.handoff_center_infos.extend(infos)
                return PlanningResult(False, stage, completed, info=self._handoff_failure_info())
            self._record_stage_snapshot(stage)

        self.handoff_center_infos.extend(infos)
        return None

    def _left_receive_pose_candidates(self) -> list[tuple[sapien.Pose, sapien.Pose, np.ndarray, dict[str, float]]]:
        env = self.base_env
        obj_pose = env.cube.pose.sp
        obj_pos = np.asarray(obj_pose.p, dtype=np.float32)
        obj_rot = quat_to_matrix(np.asarray(obj_pose.q, dtype=np.float32))
        phone_half_size = np.asarray(getattr(env, "phone_half_size", (0.075, 0.025, 0.005)), dtype=np.float32)

        receive_mode = getattr(env, "planner_handoff_receive_mode", "topdown_center")
        left_tcp_pos = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32)
        right_tcp_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32)
        local_z_magnitude = abs(float(getattr(env, "planner_left_receive_z_offset", 0.004)))
        local = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if receive_mode == "tilted_face":
            approaching = obj_rot[:, 1].astype(np.float32)
            closing = obj_rot[:, 2].astype(np.float32)
        elif receive_mode == "side_close":
            approaching = obj_rot[:, 1].astype(np.float32)
            if float(np.dot(approaching, left_tcp_pos - obj_pos)) < 0.0:
                approaching = -approaching
            closing = obj_rot[:, 2].astype(np.float32)
        else:
            # The receive point is expressed in the phone frame. local X is the
            # 15 cm long edge, local Y is the 5 cm width, and local Z is the
            # phone thickness/screen normal. The TCP still approaches from
            # above unless phone-frame orientation is explicitly enabled.
            use_phone_frame_orientation = bool(getattr(env, "planner_left_receive_use_phone_frame_orientation", False))
            if use_phone_frame_orientation:
                approaching = obj_rot[:, 2].astype(np.float32)
                if float(approaching[2]) > 0.0:
                    approaching = -approaching
            else:
                approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            closing = obj_rot[:, 1].astype(np.float32)
            if receive_mode in ("upper_side", "side_approach"):
                fraction = float(getattr(env, "planner_upper_side_receive_fraction", 0.08))
                local[0] = float(phone_half_size[0]) * fraction

        approaching_norm = float(np.linalg.norm(approaching))
        if approaching_norm < 1e-6:
            approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            approaching = approaching / approaching_norm

        local_z_sign = 1.0 if float(np.dot(obj_rot[:, 2], -approaching)) >= 0.0 else -1.0
        local[2] = local_z_sign * local_z_magnitude

        closing = closing - approaching * float(np.dot(closing, approaching))
        closing_norm = float(np.linalg.norm(closing))
        if closing_norm < 1e-6:
            closing = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            closing = closing / closing_norm

        left_tcp_q = np.asarray(self.left_agent.tcp.pose.sp.q, dtype=np.float32)
        left_tcp_y = quat_to_matrix(left_tcp_q)[:, 1]
        if float(np.dot(closing, left_tcp_y)) < 0.0:
            closing = -closing

        default_fractions = (0.08, 0.10, 0.06, 0.12) if receive_mode in ("upper_side", "side_approach") else (0.0,)
        fractions = tuple(float(v) for v in getattr(env, "planner_left_receive_candidate_fractions", default_fractions))
        manual_fraction = float(getattr(env, "planner_upper_side_receive_fraction", 0.08)) if receive_mode in ("upper_side", "side_approach") else 0.0
        if receive_mode in ("upper_side", "side_approach") and not any(abs(v - manual_fraction) < 1e-6 for v in fractions):
            fractions = (manual_fraction,) + fractions
        primary_fraction = float(getattr(env, "planner_left_receive_primary_fraction", fractions[0] if fractions else manual_fraction))
        y_offsets = tuple(float(v) for v in getattr(env, "planner_left_receive_candidate_y_offsets", (0.0,)))

        right_local = obj_rot.T @ (right_tcp_pos - obj_pos)
        min_right_clearance = float(getattr(env, "planner_left_receive_min_right_clearance", 0.030))

        candidates: list[tuple[float, sapien.Pose, dict[str, float]]] = []
        fallback: tuple[float, sapien.Pose, dict[str, float]] | None = None
        candidate_count = 0
        feasible_count = 0
        for fraction in fractions:
            for y_offset in y_offsets:
                cand_local = local.copy()
                if receive_mode in ("upper_side", "side_approach"):
                    cand_local[0] = float(phone_half_size[0]) * fraction
                cand_local[1] = float(y_offset)
                clearance = float(np.linalg.norm((cand_local - right_local)[:2]))
                if clearance < min_right_clearance:
                    continue
                receive_pos = obj_pos + obj_rot @ cand_local
                receive_pose = self.left_agent.build_grasp_pose(approaching, closing, receive_pos)
                candidate_count += 1

                distance_cost = float(np.linalg.norm(receive_pos - left_tcp_pos))
                preferred_fraction_cost = 0.02 * abs(float(fraction) - manual_fraction)
                center_cost = 0.5 * abs(float(y_offset))
                clearance_bonus = -0.05 * min(clearance, 0.08)
                score = distance_cost + preferred_fraction_cost + center_cost + clearance_bonus
                fraction_priority = 0.0 if abs(float(fraction) - primary_fraction) < 1e-8 else 0.5
                candidate_sort_score = score + fraction_priority
                info = {
                    "fraction": float(fraction),
                    "local_x": float(cand_local[0]),
                    "local_y": float(cand_local[1]),
                    "local_z": float(cand_local[2]),
                    "local_z_sign": float(local_z_sign),
                    "world_pos": receive_pos.astype(float).tolist(),
                    "approaching": approaching.astype(float).tolist(),
                    "closing": closing.astype(float).tolist(),
                    "clearance": clearance,
                    "distance_cost": distance_cost,
                    "score": float(score),
                    "fraction_priority": float(fraction_priority),
                    "sort_score": float(candidate_sort_score),
                }
                if fallback is None or candidate_sort_score < fallback[0]:
                    fallback = (candidate_sort_score, receive_pose, info)

                dry_result = self.left.solver.move_to_pose_with_screw(receive_pose, dry_run=True)
                if dry_result == -1:
                    dry_result = self.left.solver.move_to_pose_with_RRTConnect(receive_pose, dry_run=True)
                if dry_result == -1:
                    continue
                feasible_count += 1
                candidates.append((candidate_sort_score, receive_pose, info))

        if not candidates and fallback is not None:
            candidates.append(fallback)
        if not candidates:
            receive_pos = obj_pos + obj_rot @ local
            receive_pose = self.left_agent.build_grasp_pose(approaching, closing, receive_pos)
            candidates.append(
                (
                    0.0,
                    receive_pose,
                    {
                        "fraction": float(manual_fraction),
                        "local_y": 0.0,
                        "clearance": 0.0,
                        "distance_cost": float(np.linalg.norm(np.asarray(receive_pose.p, dtype=np.float32) - left_tcp_pos)),
                        "score": 0.0,
                    },
                )
            )

        candidates.sort(key=lambda item: item[0])
        pre_receive_distance = float(getattr(env, "planner_left_pre_receive_distance", 0.100))
        result: list[tuple[sapien.Pose, sapien.Pose, np.ndarray, dict[str, float]]] = []
        for score, receive_pose, info in candidates:
            selected_info = dict(info)
            selected_info["candidate_count"] = float(candidate_count)
            selected_info["feasible_count"] = float(feasible_count)
            selected_info["used_fallback"] = float(feasible_count == 0)
            selected_info["sort_score"] = float(score)
            if receive_mode == "side_approach":
                side_dir = obj_rot[:, 1].astype(np.float32)
                if float(np.dot(side_dir, left_tcp_pos - obj_pos)) < 0.0:
                    side_dir = -side_dir
                side_norm = float(np.linalg.norm(side_dir))
                side_dir = side_dir / side_norm if side_norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
                pre_receive_pos = np.asarray(receive_pose.p, dtype=np.float32) + side_dir * pre_receive_distance
                pre_receive = sapien.Pose(pre_receive_pos, receive_pose.q)
            else:
                pre_receive = receive_pose * sapien.Pose([0.0, 0.0, -pre_receive_distance])
            result.append((pre_receive, receive_pose, np.asarray(receive_pose.q, dtype=np.float32), selected_info))
        return result

    def _left_receive_poses(self) -> tuple[sapien.Pose, sapien.Pose, np.ndarray]:
        candidates = self._left_receive_pose_candidates()
        left_pre_receive, left_receive, receive_q, selected_info = candidates[0]
        self.last_left_receive_info = dict(selected_info)
        return left_pre_receive, left_receive, receive_q

    def _right_receive_pose_candidates(self) -> list[tuple[sapien.Pose, sapien.Pose, np.ndarray, dict[str, float]]]:
        env = self.base_env
        obj_pose = env.cube.pose.sp
        obj_pos = np.asarray(obj_pose.p, dtype=np.float32)
        obj_rot = quat_to_matrix(np.asarray(obj_pose.q, dtype=np.float32))
        phone_half_size = np.asarray(getattr(env, "phone_half_size", (0.075, 0.025, 0.005)), dtype=np.float32)

        receive_mode = getattr(env, "planner_handoff_receive_mode", "topdown_center")
        left_tcp_pos = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32)
        right_tcp_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32)
        if receive_mode == "tilted_face":
            approaching = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            closing = obj_rot[:, 2].astype(np.float32)
        elif receive_mode == "side_close":
            approaching = obj_rot[:, 1].astype(np.float32)
            if float(np.dot(approaching, right_tcp_pos - obj_pos)) < 0.0:
                approaching = -approaching
            closing = obj_rot[:, 2].astype(np.float32)
        else:
            use_phone_frame_orientation = bool(getattr(env, "planner_right_receive_use_phone_frame_orientation", False))
            if use_phone_frame_orientation:
                approaching = obj_rot[:, 2].astype(np.float32)
                if float(approaching[2]) > 0.0:
                    approaching = -approaching
                closing = obj_rot[:, 1].astype(np.float32)
            else:
                approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                closing = obj_rot[:, 1].astype(np.float32)
                closing[2] = 0.0

        approaching_norm = float(np.linalg.norm(approaching))
        if approaching_norm < 1e-6:
            approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            approaching = approaching / approaching_norm

        closing = closing - approaching * float(np.dot(closing, approaching))
        closing_norm = float(np.linalg.norm(closing))
        if closing_norm < 1e-6:
            closing = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            closing = closing / closing_norm

        right_tcp_q = np.asarray(self.right_agent.tcp.pose.sp.q, dtype=np.float32)
        right_tcp_y = quat_to_matrix(right_tcp_q)[:, 1]
        if receive_mode != "tilted_face" and float(np.dot(closing, right_tcp_y)) < 0.0:
            closing = -closing

        default_fractions = (-0.45, 0.45, -0.30, 0.30, -0.15, 0.15, 0.0)
        fractions = tuple(
            float(v)
            for v in getattr(env, "planner_right_receive_candidate_fractions", default_fractions)
        )
        manual_fraction = getattr(env, "planner_right_upper_side_receive_fraction", None)
        preferred = float(manual_fraction) if manual_fraction is not None else None
        if preferred is not None:
            fractions = (preferred,) + tuple(v for v in fractions if abs(v - preferred) > 1e-6)
        primary_fraction = float(getattr(env, "planner_right_receive_primary_fraction", fractions[0] if fractions else (preferred if preferred is not None else 0.0)))

        base_y_offsets = tuple(
            float(v)
            for v in getattr(env, "planner_right_receive_candidate_y_offsets", (0.0,))
        )
        manual_y_offset = float(getattr(env, "planner_right_receive_y_offset", 0.0))
        if abs(manual_y_offset) > 1e-8 and not any(abs(v - manual_y_offset) < 1e-8 for v in base_y_offsets):
            base_y_offsets = (manual_y_offset,) + base_y_offsets
        retry_y_offsets = tuple(
            float(v)
            for v in getattr(env, "planner_right_receive_retry_y_offsets", base_y_offsets)
        )
        y_offsets = base_y_offsets + tuple(
            v for v in retry_y_offsets if not any(abs(v - base_v) < 1e-8 for base_v in base_y_offsets)
        )

        default_z_offset = float(getattr(env, "planner_right_receive_z_offset", getattr(env, "planner_left_receive_z_offset", 0.004)))
        base_z_offsets = tuple(float(v) for v in getattr(env, "planner_right_receive_candidate_z_offsets", (default_z_offset,)))
        if not any(abs(v - default_z_offset) < 1e-8 for v in base_z_offsets):
            base_z_offsets = (default_z_offset,) + base_z_offsets
        retry_z_offsets = tuple(float(v) for v in getattr(env, "planner_right_receive_retry_z_offsets", base_z_offsets))
        z_offsets = base_z_offsets + tuple(
            v for v in retry_z_offsets if not any(abs(v - base_v) < 1e-8 for base_v in base_z_offsets)
        )
        local_z_sign = 1.0 if float(np.dot(obj_rot[:, 2], -approaching)) >= 0.0 else -1.0

        left_local = obj_rot.T @ (left_tcp_pos - obj_pos)
        min_left_clearance = float(getattr(env, "planner_right_receive_min_left_clearance", 0.030))
        pre_receive_distance = float(getattr(env, "planner_right_pre_receive_distance", getattr(env, "planner_left_pre_receive_distance", 0.100)))

        candidates: list[tuple[float, sapien.Pose, dict[str, float]]] = []
        fallback: tuple[float, sapien.Pose, dict[str, float]] | None = None
        candidate_count = 0
        feasible_count = 0
        for fraction in fractions:
            for y_offset in y_offsets:
                for z_offset in z_offsets:
                    local_z_offset = local_z_sign * float(z_offset)
                    local = np.array(
                        [
                            float(phone_half_size[0]) * fraction,
                            y_offset,
                            local_z_offset,
                        ],
                        dtype=np.float32,
                    )
                    clearance = float(np.linalg.norm((local - left_local)[:2]))
                    if clearance < min_left_clearance:
                        continue
                    receive_pos = obj_pos + obj_rot @ local
                    receive_pose = self.right_agent.build_grasp_pose(approaching, closing, receive_pos)
                    candidate_count += 1

                    distance_cost = float(np.linalg.norm(receive_pos - right_tcp_pos))
                    preferred_fraction_cost = 0.02 * abs(float(fraction) - float(preferred if preferred is not None else 0.0))
                    y_cost = 0.5 * abs(float(y_offset))
                    z_cost = 0.4 * abs(float(z_offset) - default_z_offset)
                    clearance_bonus = -0.05 * min(clearance, 0.08)
                    score = distance_cost + preferred_fraction_cost + y_cost + z_cost + clearance_bonus
                    fraction_priority = 0.0 if abs(float(fraction) - primary_fraction) < 1e-8 else 0.5
                    retry_priority = 0.0 if (
                        any(abs(float(y_offset) - base_v) < 1e-8 for base_v in base_y_offsets)
                        and any(abs(float(z_offset) - base_v) < 1e-8 for base_v in base_z_offsets)
                    ) else 1.0
                    candidate_sort_score = score + retry_priority + fraction_priority
                    info = {
                        "fraction": float(fraction),
                        "local_x": float(local[0]),
                        "local_y": float(y_offset),
                        "local_z": float(local[2]),
                        "local_z_sign": float(local_z_sign),
                        "z_offset": float(z_offset),
                        "world_pos": receive_pos.astype(float).tolist(),
                        "approaching": approaching.astype(float).tolist(),
                        "closing": closing.astype(float).tolist(),
                        "clearance": clearance,
                        "distance_cost": distance_cost,
                        "score": float(score),
                        "fraction_priority": float(fraction_priority),
                        "retry_priority": float(retry_priority),
                        "sort_score": float(candidate_sort_score),
                    }
                    if fallback is None or candidate_sort_score < fallback[0]:
                        fallback = (candidate_sort_score, receive_pose, info)

                    dry_result = self.right.solver.move_to_pose_with_screw(receive_pose, dry_run=True)
                    if dry_result == -1:
                        dry_result = self.right.solver.move_to_pose_with_RRTConnect(receive_pose, dry_run=True)
                    if dry_result == -1:
                        continue
                    feasible_count += 1
                    candidates.append((candidate_sort_score, receive_pose, info))

        if not candidates and fallback is not None:
            candidates.append(fallback)
        if not candidates:
            local_z_offset = local_z_sign * default_z_offset
            receive_pose = self.right_agent.build_grasp_pose(
                approaching,
                closing,
                obj_pos + obj_rot @ np.array([0.0, 0.0, local_z_offset], dtype=np.float32),
            )
            candidates.append(
                (
                    0.0,
                    receive_pose,
                    {
                        "fraction": 0.0,
                        "local_y": 0.0,
                        "local_z": float(local_z_offset),
                        "clearance": 0.0,
                        "distance_cost": float(np.linalg.norm(np.asarray(receive_pose.p, dtype=np.float32) - right_tcp_pos)),
                        "score": 0.0,
                    },
                )
            )

        candidates.sort(key=lambda item: item[0])
        result: list[tuple[sapien.Pose, sapien.Pose, np.ndarray, dict[str, float]]] = []
        for score, receive_pose, info in candidates:
            selected_info = dict(info)
            selected_info["candidate_count"] = float(candidate_count)
            selected_info["feasible_count"] = float(feasible_count)
            selected_info["used_fallback"] = float(feasible_count == 0)
            selected_info["sort_score"] = float(score)
            if receive_mode == "side_approach":
                side_dir = obj_rot[:, 1].astype(np.float32)
                if float(np.dot(side_dir, right_tcp_pos - obj_pos)) < 0.0:
                    side_dir = -side_dir
                side_norm = float(np.linalg.norm(side_dir))
                side_dir = side_dir / side_norm if side_norm > 1e-6 else np.array([0.0, -1.0, 0.0], dtype=np.float32)
                pre_receive_pos = np.asarray(receive_pose.p, dtype=np.float32) + side_dir * pre_receive_distance
                pre_receive = sapien.Pose(pre_receive_pos, receive_pose.q)
            else:
                pre_receive = receive_pose * sapien.Pose([0.0, 0.0, -pre_receive_distance])
            result.append((pre_receive, receive_pose, np.asarray(receive_pose.q, dtype=np.float32), selected_info))
        return result

    def _right_receive_poses(self) -> tuple[sapien.Pose, sapien.Pose, np.ndarray]:
        candidates = self._right_receive_pose_candidates()
        right_pre_receive, right_receive, receive_q, selected_info = candidates[0]
        self.last_right_receive_candidate = dict(selected_info)
        return right_pre_receive, right_receive, receive_q

    def _right_receive_pose_from_info(
        self,
        candidate_info: dict[str, Any],
    ) -> tuple[sapien.Pose, np.ndarray, dict[str, Any]]:
        env = self.base_env
        obj_pose = env.cube.pose.sp
        obj_pos = np.asarray(obj_pose.p, dtype=np.float32)
        obj_rot = quat_to_matrix(np.asarray(obj_pose.q, dtype=np.float32))
        receive_mode = getattr(env, "planner_handoff_receive_mode", "topdown_center")

        if receive_mode == "tilted_face":
            approaching = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            closing = obj_rot[:, 2].astype(np.float32)
        elif receive_mode == "side_close":
            tcp_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32)
            approaching = obj_rot[:, 1].astype(np.float32)
            if float(np.dot(approaching, tcp_pos - obj_pos)) < 0.0:
                approaching = -approaching
            closing = obj_rot[:, 2].astype(np.float32)
        else:
            use_phone_frame_orientation = bool(getattr(env, "planner_right_receive_use_phone_frame_orientation", False))
            if use_phone_frame_orientation:
                approaching = obj_rot[:, 2].astype(np.float32)
                if float(approaching[2]) > 0.0:
                    approaching = -approaching
                closing = obj_rot[:, 1].astype(np.float32)
            else:
                approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                closing = obj_rot[:, 1].astype(np.float32)
                closing[2] = 0.0

        approaching_norm = float(np.linalg.norm(approaching))
        if approaching_norm < 1e-6:
            approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            approaching = approaching / approaching_norm

        closing = closing - approaching * float(np.dot(closing, approaching))
        closing_norm = float(np.linalg.norm(closing))
        if closing_norm < 1e-6:
            closing = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            closing = closing / closing_norm

        right_tcp_q = np.asarray(self.right_agent.tcp.pose.sp.q, dtype=np.float32)
        right_tcp_y = quat_to_matrix(right_tcp_q)[:, 1]
        if receive_mode != "tilted_face" and float(np.dot(closing, right_tcp_y)) < 0.0:
            closing = -closing

        target_local = np.array(
            [
                float(candidate_info.get("local_x", 0.0)),
                float(candidate_info.get("local_y", 0.0)),
                float(candidate_info.get("local_z", 0.0)),
            ],
            dtype=np.float32,
        )
        receive_pos = obj_pos + obj_rot @ target_local
        receive_pose = self.right_agent.build_grasp_pose(approaching, closing, receive_pos)
        info = {
            "target_local": target_local.astype(float).tolist(),
            "target_world": receive_pos.astype(float).tolist(),
            "approaching": approaching.astype(float).tolist(),
            "closing": closing.astype(float).tolist(),
        }
        return receive_pose, np.asarray(receive_pose.q, dtype=np.float32), info

    def _run_right_receive_closed_loop(
        self,
        candidate_info: dict[str, Any],
        completed: list[str],
        suffix: str,
    ) -> tuple[bool, str | None, sapien.Pose, np.ndarray, list[dict[str, Any]]]:
        receive_pose, receive_q, pose_info = self._right_receive_pose_from_info(candidate_info)
        infos: list[dict[str, Any]] = []
        if not bool(getattr(self.base_env, "planner_right_receive_closed_loop_enabled", True)):
            return True, None, receive_pose, receive_q, infos

        attempts = int(getattr(self.base_env, "planner_right_receive_closed_loop_attempts", 3))
        tolerance = float(getattr(self.base_env, "planner_right_receive_closed_loop_tolerance", 0.004))
        orientation_tolerance = float(getattr(self.base_env, "planner_right_receive_closed_loop_orientation_tolerance_deg", 8.0))
        refine_steps = int(getattr(self.base_env, "planner_right_receive_closed_loop_refine_steps", 4))
        target_local = np.asarray(pose_info["target_local"], dtype=np.float32)
        last_stage = "right_receive_closed_loop"

        for attempt in range(max(1, attempts)):
            obj_pose = self.base_env.cube.pose.sp
            obj_pos = np.asarray(obj_pose.p, dtype=np.float32)
            obj_rot = quat_to_matrix(np.asarray(obj_pose.q, dtype=np.float32))
            tcp_pose = self.right_agent.tcp.pose.sp
            tcp_pos = np.asarray(tcp_pose.p, dtype=np.float32)
            actual_local = obj_rot.T @ (tcp_pos - obj_pos)
            local_error = target_local - actual_local
            error_norm = float(np.linalg.norm(local_error))
            receive_pose, receive_q, pose_info = self._right_receive_pose_from_info(candidate_info)
            actual_tcp_rot = quat_to_matrix(np.asarray(tcp_pose.q, dtype=np.float32))
            target_tcp_rot = quat_to_matrix(np.asarray(receive_pose.q, dtype=np.float32))
            closing_axis_error = vector_angle_deg(actual_tcp_rot[:, 1], target_tcp_rot[:, 1])
            approach_axis_error = vector_angle_deg(actual_tcp_rot[:, 2], target_tcp_rot[:, 2])
            orientation_error = max(closing_axis_error, approach_axis_error)
            loop_info = {
                "attempt": float(attempt),
                "target_local": target_local.astype(float).tolist(),
                "actual_local": actual_local.astype(float).tolist(),
                "local_error": local_error.astype(float).tolist(),
                "local_error_norm": error_norm,
                "tolerance": tolerance,
                "closing_axis_error_deg": closing_axis_error,
                "approach_axis_error_deg": approach_axis_error,
                "orientation_error_deg": orientation_error,
                "orientation_tolerance_deg": orientation_tolerance,
                "target_world": pose_info["target_world"],
            }
            infos.append(loop_info)
            if error_norm <= tolerance and orientation_error <= orientation_tolerance:
                break

            last_stage = f"right_receive_closed_loop{suffix}_{attempt + 1}"
            if not self.right._run_stage(
                last_stage,
                receive_pose,
                completed,
                self.right._refine(refine_steps),
            ):
                return False, last_stage, receive_pose, receive_q, infos
            self._record_stage_snapshot(last_stage)

        return True, None, receive_pose, receive_q, infos

    def _right_object_pose_aligned_tcp_q(self) -> np.ndarray:
        return self._object_pose_aligned_tcp_q("right", self.right_agent)

    def _run_right_calibrate_and_insert(
        self,
        completed: list[str],
        goal_pos: np.ndarray,
        receive_q: np.ndarray,
    ) -> PlanningResult | None:
        target_angle = float(getattr(self.base_env, "planner_insert_angle_deg", 90.0))
        handoff_angle = float(getattr(self.base_env, "planner_handoff_angle_deg", 45.0))
        delta_angle = target_angle - handoff_angle
        insert_q = receive_q
        pre_insert_height = float(getattr(self.base_env, "planner_right_pre_insert_height", 0.120))
        insert_heights = tuple(
            float(v)
            for v in getattr(
                self.base_env,
                "planner_right_pose_guided_insert_heights",
                (0.080, 0.040, 0.015, 0.0),
            )
        )
        use_regularized_insert = getattr(self.base_env, "planner_use_regularized_insert", True)

        tcp_minus_obj = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32) - self._current_obj_pos()
        pre_insert_pos = goal_pos + tcp_minus_obj + z_offset(pre_insert_height)
        if not self.right._run_stage(
            "right_pre_insert_45deg",
            sapien.Pose(pre_insert_pos, receive_q),
            completed,
            self.right._refine(4),
            regularized=use_regularized_insert,
        ):
            return PlanningResult(False, "right_pre_insert_45deg", completed)
        self._record_stage_snapshot("right_pre_insert_45deg")

        if abs(delta_angle) > 1.0:
            calibrate_z_offset = float(getattr(self.base_env, "planner_right_calibrate_z_offset", getattr(self.base_env, "planner_left_calibrate_z_offset", 0.030)))
            rotate_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(calibrate_z_offset)
            calibration_alphas = (1.0,) if bool(getattr(self.base_env, "planner_insert_calibrate_single_step", False)) else (0.5, 1.0)
            for alpha in calibration_alphas:
                q = quat_mul(self._world_y_rotation_quat(delta_angle * alpha), receive_q)
                stage = f"right_calibrate_{int(round(handoff_angle + delta_angle * alpha))}"
                pose = sapien.Pose(rotate_pos, q)
                if (
                    self.return_home
                    and bool(getattr(self.base_env, "planner_idle_return_home_during_insert", False))
                    and "left" not in self.idle_returned_home_sides
                ):
                    ok = self._run_parallel_insert_lift_and_idle_home(
                        release_side="left",
                        release_solver=self.left,
                        release_agent=self.left_agent,
                        insert_lift_stage=stage,
                        insert_solver=self.right,
                        insert_agent=self.right_agent,
                        insert_lift_pose=pose,
                        insert_lift_refine=self.right._refine(5),
                        completed=completed,
                    )
                    if ok:
                        self.idle_returned_home_sides.add("left")
                else:
                    ok = self.right._run_stage(
                        stage,
                        pose,
                        completed,
                        self.right._refine(5),
                    )
                    if ok:
                        self._record_stage_snapshot(stage)
                if not ok:
                    return PlanningResult(False, stage, completed)
            insert_q = quat_mul(self._world_y_rotation_quat(delta_angle), receive_q)
        else:
            completed.append(f"right_calibrate_{int(round(target_angle))}_skipped")
            self._record_stage_snapshot(completed[-1])

        ok, insert_q = self._align_object_pose_before_insert("right", self.right, self.right_agent, completed, insert_q)
        if not ok:
            return PlanningResult(False, "right_align_object_pose", completed)

        ok, insert_q = self._ensure_object_ready_for_insert("right", self.right, self.right_agent, completed, goal_pos, insert_q)
        if not ok:
            return PlanningResult(
                False,
                "right_insert_readiness",
                completed,
                info={
                    "planner_object_align_infos": self.object_align_infos,
                    "planner_insert_readiness_infos": self.insert_readiness_infos,
                    "planner_right_insert_readiness_info": self.last_right_insert_readiness_info,
                },
            )

        for index, height in enumerate(insert_heights):
            if bool(getattr(self.base_env, "planner_right_align_object_pose_during_insert", True)):
                insert_q = self._object_pose_aligned_tcp_q("right", self.right_agent)
            tcp_minus_obj = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32) - self._current_obj_pos()
            target_obj_pos = goal_pos + z_offset(height)
            target_tcp_pos = target_obj_pos + tcp_minus_obj
            stage = "right_insert" if index == len(insert_heights) - 1 else f"right_insert_pose_guided_{index}_{int(round(height * 1000))}mm"
            refine_steps = (
                int(getattr(self.base_env, "planner_insert_final_refine_steps", 14))
                if index == len(insert_heights) - 1
                else int(getattr(self.base_env, "planner_insert_intermediate_refine_steps", 6))
            )
            refine = self.right._refine(refine_steps)
            if not self.right._run_stage(
                stage,
                sapien.Pose(target_tcp_pos, insert_q),
                completed,
                refine,
                regularized=use_regularized_insert,
            ):
                return PlanningResult(False, stage, completed)
            self._record_stage_snapshot(stage)
        return None

    def _run_left_calibrate_and_insert(
        self,
        completed: list[str],
        goal_pos: np.ndarray,
        receive_q: np.ndarray,
    ) -> PlanningResult | None:
        target_angle = float(getattr(self.base_env, "planner_insert_angle_deg", 90.0))
        handoff_angle = float(getattr(self.base_env, "planner_handoff_angle_deg", 45.0))
        delta_angle = target_angle - handoff_angle
        insert_q = receive_q
        pre_insert_height = float(getattr(self.base_env, "planner_left_pre_insert_height", 0.120))
        insert_heights = tuple(
            float(v)
            for v in getattr(
                self.base_env,
                "planner_left_pose_guided_insert_heights",
                getattr(self.base_env, "planner_right_pose_guided_insert_heights", (0.080, 0.040, 0.015, 0.0)),
            )
        )
        use_regularized_insert = getattr(self.base_env, "planner_use_regularized_insert", True)
        tcp_minus_obj = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32) - self._current_obj_pos()
        pre_insert_pos = goal_pos + tcp_minus_obj + z_offset(pre_insert_height)
        if not self.left._run_stage(
            "left_pre_insert_45deg",
            sapien.Pose(pre_insert_pos, receive_q),
            completed,
            self.left._refine(4),
            regularized=use_regularized_insert,
        ):
            return PlanningResult(False, "left_pre_insert_45deg", completed)
        self._record_stage_snapshot("left_pre_insert_45deg")

        if abs(delta_angle) > 1.0:
            calibrate_z_offset = float(getattr(self.base_env, "planner_left_calibrate_z_offset", 0.030))
            rotate_pos = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(calibrate_z_offset)
            calibration_alphas = (1.0,) if bool(getattr(self.base_env, "planner_insert_calibrate_single_step", False)) else (0.5, 1.0)
            for alpha in calibration_alphas:
                q = quat_mul(self._world_y_rotation_quat(delta_angle * alpha), receive_q)
                stage = f"left_calibrate_{int(round(handoff_angle + delta_angle * alpha))}"
                pose = sapien.Pose(rotate_pos, q)
                if (
                    self.return_home
                    and bool(getattr(self.base_env, "planner_idle_return_home_during_insert", False))
                    and "right" not in self.idle_returned_home_sides
                ):
                    ok = self._run_parallel_insert_lift_and_idle_home(
                        release_side="right",
                        release_solver=self.right,
                        release_agent=self.right_agent,
                        insert_lift_stage=stage,
                        insert_solver=self.left,
                        insert_agent=self.left_agent,
                        insert_lift_pose=pose,
                        insert_lift_refine=self.left._refine(5),
                        completed=completed,
                    )
                    if ok:
                        self.idle_returned_home_sides.add("right")
                else:
                    ok = self.left._run_stage(
                        stage,
                        pose,
                        completed,
                        self.left._refine(5),
                    )
                    if ok:
                        self._record_stage_snapshot(stage)
                if not ok:
                    return PlanningResult(False, stage, completed)
            insert_q = quat_mul(self._world_y_rotation_quat(delta_angle), receive_q)
        else:
            completed.append(f"left_calibrate_{int(round(target_angle))}_skipped")

        ok, insert_q = self._align_object_pose_before_insert("left", self.left, self.left_agent, completed, insert_q)
        if not ok:
            return PlanningResult(False, "left_align_object_pose", completed)

        ok, insert_q = self._ensure_object_ready_for_insert("left", self.left, self.left_agent, completed, goal_pos, insert_q)
        if not ok:
            return PlanningResult(
                False,
                "left_insert_readiness",
                completed,
                info={
                    "planner_object_align_infos": self.object_align_infos,
                    "planner_insert_readiness_infos": self.insert_readiness_infos,
                    "planner_left_insert_readiness_info": self.last_left_insert_readiness_info,
                },
            )

        for index, height in enumerate(insert_heights):
            if bool(getattr(self.base_env, "planner_left_align_object_pose_during_insert", getattr(self.base_env, "planner_right_align_object_pose_during_insert", True))):
                insert_q = self._object_pose_aligned_tcp_q("left", self.left_agent)
            tcp_minus_obj = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32) - self._current_obj_pos()
            target_obj_pos = goal_pos + z_offset(height)
            target_tcp_pos = target_obj_pos + tcp_minus_obj
            stage = "left_insert" if index == len(insert_heights) - 1 else f"left_insert_pose_guided_{index}_{int(round(height * 1000))}mm"
            refine_steps = (
                int(getattr(self.base_env, "planner_insert_final_refine_steps", 14))
                if index == len(insert_heights) - 1
                else int(getattr(self.base_env, "planner_insert_intermediate_refine_steps", 6))
            )
            refine = self.left._refine(refine_steps)
            if not self.left._run_stage(stage, sapien.Pose(target_tcp_pos, insert_q), completed, refine, regularized=use_regularized_insert):
                return PlanningResult(False, stage, completed)
            self._record_stage_snapshot(stage)
        return None

    def _pick_left_flip_handoff_right_insert(
        self,
        obj_pos: np.ndarray,
        goal_pos: np.ndarray,
        close_steps: int,
        open_steps: int,
    ) -> PlanningResult:
        completed: list[str] = []
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        pre_grasp, grasp, lift_flat, tcp_q, insert_q = self._left_phone_grasp_poses(obj_pos)

        if not self.left._run_stage("left_pre_grasp", pre_grasp, completed, self.left._refine(int(getattr(self.base_env, "planner_pre_grasp_refine_steps", 2)))):
            return PlanningResult(False, "left_pre_grasp", completed)
        if not self.left._run_local_cartesian_stage("left_grasp", grasp, completed, check_object_static=True):
            return PlanningResult(False, "left_grasp", completed, info=self._local_cartesian_info())

        self._close_gripper_until_grasp(
            "left",
            self.left,
            self.left_agent,
            completed,
            "left_close_gripper",
            self.left._refine(close_steps),
        )

        if not self.left._run_local_cartesian_stage("left_lift_flat", lift_flat, completed):
            return PlanningResult(False, "left_lift_flat", completed, info=self._local_cartesian_info())
        self._record_stage_snapshot("left_lift_flat")

        failed = self._move_held_object_to_handoff_center("left", self.left, self.left_agent, completed)
        if failed is not None:
            return failed

        target_angle = float(getattr(self.base_env, "planner_insert_angle_deg", 90.0))
        handoff_angle = float(getattr(self.base_env, "planner_handoff_angle_deg", 45.0))
        alpha = float(np.clip(handoff_angle / max(abs(target_angle), 1e-6), 0.05, 1.0))
        rotate_z_offset = float(getattr(self.base_env, "planner_left_flip_z_offset", getattr(self.base_env, "planner_right_flip_z_offset", 0.040)))
        rotate_pos = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(rotate_z_offset)
        if not self.left._run_rotation_stage(
            f"left_flip_handoff_{int(round(handoff_angle))}",
            rotate_pos,
            tcp_q,
            insert_q,
            (alpha,),
            completed,
        ):
            return PlanningResult(False, "left_flip_handoff", completed)
        self._record_stage_snapshot(f"left_flip_handoff_{int(round(handoff_angle))}")
        pre_receive_refine = int(getattr(self.base_env, "planner_pre_receive_refine_steps", 4))
        receive_refine = int(getattr(self.base_env, "planner_receive_refine_steps", 10))
        max_receive_attempts = int(getattr(self.base_env, "planner_right_receive_retry_count", 1))
        receive_q: np.ndarray | None = None
        right_receive: sapien.Pose | None = None
        receive_attempt_infos: list[dict[str, Any]] = []
        right_receive_closed_loop_infos: list[dict[str, Any]] = []
        receive_candidates = self._right_receive_pose_candidates()[:max(1, max_receive_attempts)]
        for attempt_index, (right_pre_receive, candidate_receive, candidate_q, candidate_info) in enumerate(receive_candidates):
            self.last_right_receive_candidate = dict(candidate_info)
            self.last_right_receive_candidate["attempt_index"] = float(attempt_index)
            suffix = "" if attempt_index == 0 else f"_retry_{attempt_index}"

            pre_stage = f"right_pre_receive{suffix}"
            if not self.right._run_stage(pre_stage, right_pre_receive, completed, self.right._refine(pre_receive_refine)):
                return PlanningResult(False, pre_stage, completed)
            self._record_stage_snapshot(pre_stage)

            # Mirror the left-hand receive flow: after the arm reaches pre-receive,
            # re-read the phone pose and recompute the receive target from the live
            # object frame. The left holder can slightly move the phone while the
            # receiver approaches, and using the stale pre-receive candidate makes
            # the right hand need several correction stages.
            live_candidates = self._right_receive_pose_candidates()[:max(1, max_receive_attempts)]
            if live_candidates:
                live_index = min(attempt_index, len(live_candidates) - 1)
                _, candidate_receive, candidate_q, candidate_info = live_candidates[live_index]
                self.last_right_receive_candidate = dict(candidate_info)
                self.last_right_receive_candidate["attempt_index"] = float(attempt_index)
                self.last_right_receive_candidate["recomputed_after_pre_receive"] = True

            receive_stage = f"right_receive{suffix}"
            if not self.right._run_stage(receive_stage, candidate_receive, completed, self.right._refine(receive_refine)):
                return PlanningResult(False, receive_stage, completed)
            self._record_stage_snapshot(receive_stage)

            closed_loop_ok, closed_loop_stage, candidate_receive, candidate_q, closed_loop_infos = self._run_right_receive_closed_loop(
                self.last_right_receive_candidate,
                completed,
                suffix,
            )
            self.last_right_receive_candidate["closed_loop_infos"] = closed_loop_infos
            right_receive_closed_loop_infos.extend(closed_loop_infos)
            self.right_receive_closed_loop_infos = right_receive_closed_loop_infos
            if not closed_loop_ok:
                return PlanningResult(False, closed_loop_stage or "right_receive_closed_loop", completed, info=self._handoff_failure_info())

            close_stage = f"right_close_gripper{suffix}"
            self._close_gripper_until_grasp(
                "right",
                self.right,
                self.right_agent,
                completed,
                close_stage,
                self.right._refine(close_steps),
            )
            confirm_stage = f"right_confirm_handoff_grasp_before_left_release{suffix}"
            grasp_confirmed = self._confirm_handoff_grasp("right", self.right_agent, completed, confirm_stage)
            attempt_info = dict(self.last_right_receive_candidate)
            attempt_info["confirmed"] = bool(grasp_confirmed)
            receive_attempt_infos.append(attempt_info)
            if grasp_confirmed:
                right_receive = candidate_receive
                receive_q = candidate_q
                break

            if attempt_index < len(receive_candidates) - 1:
                self.right.solver.open_gripper(t=self.right._refine(open_steps))
                retry_open_stage = f"right_open_before_receive_retry_{attempt_index + 1}"
                completed.append(retry_open_stage)
                self._record_stage_snapshot(retry_open_stage)

        self.right_receive_attempt_infos = receive_attempt_infos
        self.right_receive_closed_loop_infos = right_receive_closed_loop_infos
        if receive_q is None or right_receive is None:
            return PlanningResult(False, "right_confirm_handoff_grasp_before_left_release", completed, info=self._handoff_failure_info())

        settle_steps = int(getattr(self.base_env, "planner_right_receive_settle_steps", 0))
        if settle_steps > 0:
            if not self.right._run_stage(
                "right_settle_after_receive",
                right_receive,
                completed,
                self.right._refine(settle_steps),
            ):
                return PlanningResult(False, "right_settle_after_receive", completed)
            self._record_stage_snapshot("right_settle_after_receive")

        self.left.solver.open_gripper(t=self.left._refine(open_steps))
        completed.append("left_release_handoff")
        self._record_stage_snapshot("left_release_handoff")
        if not self._confirm_handoff_grasp("right", self.right_agent, completed, "right_confirm_handoff_grasp_after_left_release"):
            return PlanningResult(False, "right_confirm_handoff_grasp_after_left_release", completed, info=self._handoff_failure_info())

        right_lift_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(
            float(getattr(self.base_env, "planner_right_handoff_lift_height", getattr(self.base_env, "planner_left_handoff_lift_height", 0.020)))
        )
        failed = self._run_post_handoff_retract_with_insert_lift(
            "left",
            completed,
            x=float(getattr(self.base_env, "planner_left_retract_after_right_handoff_x", 0.0)),
            y=float(getattr(self.base_env, "planner_left_retract_after_right_handoff_y", 0.0)),
            z=float(getattr(self.base_env, "planner_left_retract_after_right_handoff_z", 0.100)),
            insert_lift_stage="right_lift_after_handoff",
            insert_solver=self.right,
            insert_agent=self.right_agent,
            insert_lift_pose=sapien.Pose(right_lift_pos, receive_q),
            insert_lift_refine=self.right._refine(6),
            goal_pos=goal_pos,
        )
        if failed is not None:
            return failed

        failed = self._run_right_calibrate_and_insert(completed, goal_pos, receive_q)
        if failed is not None:
            return failed

        failed = self._right_open_retract_and_return(completed, open_steps)
        if failed is not None:
            return failed

        return PlanningResult(
            True,
            None,
            completed,
            info={
                "grasp_candidate_index": 0,
                "grasp_candidate_label": "two_panda_flip_handoff_right_insert",
                "two_panda_mode": "handoff",
                "planner_insert_arm": "right",
                "planner_insert_arm_mode": str(getattr(self.base_env, "planner_insert_arm_mode", "auto_by_slot")),
                "planner_slot_id": self._slot_id_for_goal(goal_pos),
                "left_agent_uid": self.left_uid,
                "right_agent_uid": self.right_uid,
                "handoff_angle_deg": handoff_angle,
                "planner_insert_angle_deg": target_angle,
                "planner_left_pre_grasp_height": float(getattr(self.base_env, "planner_left_pre_grasp_height", 0.080)),
                "planner_left_lift_height": float(getattr(self.base_env, "planner_left_lift_height", 0.160)),
                "planner_left_flip_z_offset": float(getattr(self.base_env, "planner_left_flip_z_offset", getattr(self.base_env, "planner_right_flip_z_offset", 0.040))),
                "planner_right_upper_side_receive_fraction": float(getattr(self.base_env, "planner_right_upper_side_receive_fraction", getattr(self.base_env, "planner_upper_side_receive_fraction", 0.08))),
                "planner_right_receive_y_offset": float(getattr(self.base_env, "planner_right_receive_y_offset", 0.0)),
                "planner_right_receive_z_offset": float(getattr(self.base_env, "planner_right_receive_z_offset", getattr(self.base_env, "planner_left_receive_z_offset", 0.004))),
                "planner_right_receive_candidate_fractions": list(getattr(self.base_env, "planner_right_receive_candidate_fractions", ())),
                "planner_right_receive_candidate_y_offsets": list(getattr(self.base_env, "planner_right_receive_candidate_y_offsets", ())),
                "planner_right_receive_min_left_clearance": float(getattr(self.base_env, "planner_right_receive_min_left_clearance", 0.030)),
                "planner_right_handoff_lift_height": float(getattr(self.base_env, "planner_right_handoff_lift_height", getattr(self.base_env, "planner_left_handoff_lift_height", 0.020))),
                "planner_right_pre_receive_distance": float(getattr(self.base_env, "planner_right_pre_receive_distance", getattr(self.base_env, "planner_left_pre_receive_distance", 0.100))),
                "planner_right_receive_candidate": getattr(self, "last_right_receive_candidate", None),
                "planner_right_receive_attempt_infos": getattr(self, "right_receive_attempt_infos", []),
                "planner_right_receive_closed_loop_infos": getattr(self, "right_receive_closed_loop_infos", []),
                "planner_right_receive_retry_count": int(getattr(self.base_env, "planner_right_receive_retry_count", 1)),
                "planner_right_receive_candidate_z_offsets": list(getattr(self.base_env, "planner_right_receive_candidate_z_offsets", ())),
                "planner_right_receive_use_phone_frame_orientation": bool(getattr(self.base_env, "planner_right_receive_use_phone_frame_orientation", False)),
                "planner_right_receive_settle_steps": int(getattr(self.base_env, "planner_right_receive_settle_steps", 0)),
                "planner_left_retract_after_right_handoff_x": float(getattr(self.base_env, "planner_left_retract_after_right_handoff_x", 0.0)),
                "planner_left_retract_after_right_handoff_y": float(getattr(self.base_env, "planner_left_retract_after_right_handoff_y", 0.0)),
                "planner_left_retract_after_right_handoff_z": float(getattr(self.base_env, "planner_left_retract_after_right_handoff_z", 0.100)),
                "planner_post_handoff_retract_refine_steps": int(getattr(self.base_env, "planner_post_handoff_retract_refine_steps", 4)),
                "planner_stage_snapshots": self.stage_snapshots,
                "planner_state_close_infos": self.state_close_infos,
                "planner_local_cartesian_infos": self.left.local_cartesian_infos + self.right.local_cartesian_infos,
                "planner_handoff_confirm_infos": self.handoff_confirm_infos,
                "planner_handoff_center_xy": list(getattr(self.base_env, "planner_handoff_center_xy", ())),
                "planner_handoff_center_infos": self.handoff_center_infos,
                "planner_object_align_infos": self.object_align_infos,
                "planner_insert_readiness_infos": self.insert_readiness_infos,
                "planner_right_insert_readiness_info": self.last_right_insert_readiness_info,
                "planner_receive_z_offset_frame": "phone_local_z",
                "planner_right_align_object_pose_before_insert": bool(getattr(self.base_env, "planner_right_align_object_pose_before_insert", True)),
                "planner_right_object_align_z_offset": float(getattr(self.base_env, "planner_right_object_align_z_offset", 0.010)),
                "planner_right_object_align_max_angle_deg": float(getattr(self.base_env, "planner_right_object_align_max_angle_deg", 20.0)),
                "planner_right_object_align_info": self.last_right_object_align_info,
                "planner_right_calibrate_z_offset": float(getattr(self.base_env, "planner_right_calibrate_z_offset", getattr(self.base_env, "planner_left_calibrate_z_offset", 0.030))),
                "planner_right_pre_insert_height": float(getattr(self.base_env, "planner_right_pre_insert_height", 0.120)),
                "planner_right_pose_guided_insert_heights": list(getattr(self.base_env, "planner_right_pose_guided_insert_heights", ())),
                "planner_right_align_object_pose_during_insert": bool(getattr(self.base_env, "planner_right_align_object_pose_during_insert", True)),
                "planner_right_post_release_lift_height": float(getattr(self.base_env, "planner_right_post_release_lift_height", 0.0)),
                "return_home": self.return_home,
            },
        )

    def _pick_flip_handoff_insert(
        self,
        obj_pos: np.ndarray,
        goal_pos: np.ndarray,
        close_steps: int,
        open_steps: int,
    ) -> PlanningResult:
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        insert_arm = self._select_insert_arm_for_slot(goal_pos)
        if insert_arm == "right":
            return self._pick_left_flip_handoff_right_insert(obj_pos, goal_pos, close_steps, open_steps)

        completed: list[str] = []
        pre_grasp, grasp, lift_flat, tcp_q, insert_q = self._right_phone_grasp_poses(obj_pos)

        if not self.right._run_stage("right_pre_grasp", pre_grasp, completed, self.right._refine(int(getattr(self.base_env, "planner_pre_grasp_refine_steps", 2)))):
            return PlanningResult(False, "right_pre_grasp", completed)
        if not self.right._run_local_cartesian_stage("right_grasp", grasp, completed, check_object_static=True):
            return PlanningResult(False, "right_grasp", completed, info=self._local_cartesian_info())

        self._close_gripper_until_grasp(
            "right",
            self.right,
            self.right_agent,
            completed,
            "right_close_gripper",
            self.right._refine(close_steps),
        )

        if not self.right._run_local_cartesian_stage("right_lift_flat", lift_flat, completed):
            return PlanningResult(False, "right_lift_flat", completed, info=self._local_cartesian_info())
        self._record_stage_snapshot("right_lift_flat")

        failed = self._move_held_object_to_handoff_center("right", self.right, self.right_agent, completed)
        if failed is not None:
            return failed

        target_angle = float(getattr(self.base_env, "planner_insert_angle_deg", 90.0))
        handoff_angle = float(getattr(self.base_env, "planner_handoff_angle_deg", 45.0))
        alpha = float(np.clip(handoff_angle / max(abs(target_angle), 1e-6), 0.05, 1.0))
        rotate_z_offset = float(getattr(self.base_env, "planner_right_flip_z_offset", 0.040))
        rotate_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(rotate_z_offset)
        if not self.right._run_rotation_stage(
            f"right_flip_handoff_{int(round(handoff_angle))}",
            rotate_pos,
            tcp_q,
            insert_q,
            (alpha,),
            completed,
        ):
            return PlanningResult(False, "right_flip_handoff", completed)
        self._record_stage_snapshot(f"right_flip_handoff_{int(round(handoff_angle))}")
        left_pre_receive, _, _ = self._left_receive_poses()
        pre_receive_refine = int(getattr(self.base_env, "planner_pre_receive_refine_steps", 4))
        receive_refine = int(getattr(self.base_env, "planner_receive_refine_steps", 10))
        if not self.left._run_stage("left_pre_receive", left_pre_receive, completed, self.left._refine(pre_receive_refine)):
            return PlanningResult(False, "left_pre_receive", completed)
        self._record_stage_snapshot("left_pre_receive")

        max_receive_attempts = int(getattr(self.base_env, "planner_left_receive_retry_count", 1))
        receive_q: np.ndarray | None = None
        left_receive_attempt_infos: list[dict[str, Any]] = []
        receive_candidates = self._left_receive_pose_candidates()[:max(1, max_receive_attempts)]
        for attempt_index, (candidate_pre_receive, candidate_receive, candidate_q, candidate_info) in enumerate(receive_candidates):
            self.last_left_receive_info = dict(candidate_info)
            self.last_left_receive_info["attempt_index"] = float(attempt_index)
            suffix = "" if attempt_index == 0 else f"_retry_{attempt_index}"

            if attempt_index > 0:
                pre_stage = f"left_pre_receive{suffix}"
                if not self.left._run_stage(pre_stage, candidate_pre_receive, completed, self.left._refine(pre_receive_refine)):
                    return PlanningResult(False, pre_stage, completed)
                self._record_stage_snapshot(pre_stage)

            receive_stage = f"left_receive{suffix}"
            if not self.left._run_stage(receive_stage, candidate_receive, completed, self.left._refine(receive_refine)):
                return PlanningResult(False, receive_stage, completed)
            self._record_stage_snapshot(receive_stage)

            close_stage = f"left_close_gripper{suffix}"
            self._close_gripper_until_grasp(
                "left",
                self.left,
                self.left_agent,
                completed,
                close_stage,
                self.left._refine(close_steps),
            )
            confirm_stage = f"left_confirm_handoff_grasp_before_right_release{suffix}"
            grasp_confirmed = self._confirm_handoff_grasp("left", self.left_agent, completed, confirm_stage)
            attempt_info = dict(self.last_left_receive_info)
            attempt_info["confirmed"] = bool(grasp_confirmed)
            left_receive_attempt_infos.append(attempt_info)
            if grasp_confirmed:
                receive_q = candidate_q
                break

            if attempt_index < len(receive_candidates) - 1:
                self.left.solver.open_gripper(t=self.left._refine(open_steps))
                retry_open_stage = f"left_open_before_receive_retry_{attempt_index + 1}"
                completed.append(retry_open_stage)
                self._record_stage_snapshot(retry_open_stage)

        self.left_receive_attempt_infos = left_receive_attempt_infos
        if receive_q is None:
            return PlanningResult(False, "left_confirm_handoff_grasp_before_right_release", completed, info=self._handoff_failure_info())
        self.right.solver.open_gripper(t=self.right._refine(open_steps))
        completed.append("right_release_handoff")
        self._record_stage_snapshot("right_release_handoff")
        if not self._confirm_handoff_grasp("left", self.left_agent, completed, "left_confirm_handoff_grasp_after_right_release"):
            return PlanningResult(False, "left_confirm_handoff_grasp_after_right_release", completed, info=self._handoff_failure_info())

        left_lift_pos = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(float(getattr(self.base_env, "planner_left_handoff_lift_height", 0.020)))
        failed = self._run_post_handoff_retract_with_insert_lift(
            "right",
            completed,
            x=float(getattr(self.base_env, "planner_right_retract_after_left_handoff_x", 0.0)),
            y=float(getattr(self.base_env, "planner_right_retract_after_left_handoff_y", 0.080)),
            z=float(getattr(self.base_env, "planner_right_retract_after_left_handoff_z", 0.080)),
            insert_lift_stage="left_lift_after_handoff",
            insert_solver=self.left,
            insert_agent=self.left_agent,
            insert_lift_pose=sapien.Pose(left_lift_pos, receive_q),
            insert_lift_refine=self.left._refine(6),
            goal_pos=goal_pos,
        )
        if failed is not None:
            return failed

        failed = self._run_left_calibrate_and_insert(completed, goal_pos, receive_q)
        if failed is not None:
            return failed

        self.left.solver.open_gripper(t=self.left._refine(open_steps))
        completed.append("left_open_gripper")
        self._record_stage_snapshot("left_open_gripper")

        post_release_lift = float(getattr(self.base_env, "planner_left_post_release_lift_height", 0.0))
        if post_release_lift > 0.0:
            left_retract_pos = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(post_release_lift)
            left_retract_q = np.asarray(self.left_agent.tcp.pose.sp.q, dtype=np.float32)
            if not self.left._run_stage(
                "left_retract_after_release",
                sapien.Pose(left_retract_pos, left_retract_q),
                completed,
                self.left._refine(4),
            ):
                return PlanningResult(False, "left_retract_after_release", completed)

        if self.return_home:
            idle_returned = "right" in getattr(self, "idle_returned_home_sides", set())
            if not idle_returned and not self.right.move_to_qpos(self.right_initial_qpos, refine_steps=self.right._refine(4)):
                return PlanningResult(False, "right_return_home", completed)
            if not idle_returned:
                completed.append("right_return_home")
            if not self.left.move_to_qpos(self.left_initial_qpos, refine_steps=self.left._refine(4)):
                return PlanningResult(False, "left_return_home", completed)
            completed.append("left_return_home")

        return PlanningResult(
            True,
            None,
            completed,
            info={
                "grasp_candidate_index": 0,
                "grasp_candidate_label": "two_panda_flip_handoff_left_insert",
                "two_panda_mode": "handoff",
                "planner_insert_arm": "left",
                "planner_insert_arm_mode": str(getattr(self.base_env, "planner_insert_arm_mode", "auto_by_slot")),
                "planner_slot_id": self._slot_id_for_goal(goal_pos),
                "left_agent_uid": self.left_uid,
                "right_agent_uid": self.right_uid,
                "handoff_angle_deg": handoff_angle,
                "planner_insert_angle_deg": target_angle,
                "planner_right_pre_grasp_height": float(getattr(self.base_env, "planner_right_pre_grasp_height", 0.080)),
                "planner_right_lift_height": float(getattr(self.base_env, "planner_right_lift_height", 0.160)),
                "planner_right_flip_z_offset": float(getattr(self.base_env, "planner_right_flip_z_offset", 0.040)),
                "planner_right_pre_insert_height": float(getattr(self.base_env, "planner_right_pre_insert_height", 0.120)),
                "planner_right_pose_guided_insert_heights": list(getattr(self.base_env, "planner_right_pose_guided_insert_heights", ())),
                "planner_right_align_object_pose_during_insert": bool(getattr(self.base_env, "planner_right_align_object_pose_during_insert", True)),
                "planner_right_post_release_lift_height": float(getattr(self.base_env, "planner_right_post_release_lift_height", 0.0)),
                "planner_right_retract_after_left_handoff_x": float(getattr(self.base_env, "planner_right_retract_after_left_handoff_x", 0.0)),
                "planner_right_retract_after_left_handoff_y": float(getattr(self.base_env, "planner_right_retract_after_left_handoff_y", 0.080)),
                "planner_right_retract_after_left_handoff_z": float(getattr(self.base_env, "planner_right_retract_after_left_handoff_z", 0.080)),
                "planner_post_handoff_retract_refine_steps": int(getattr(self.base_env, "planner_post_handoff_retract_refine_steps", 4)),
                "planner_left_receive_z_offset": float(getattr(self.base_env, "planner_left_receive_z_offset", 0.004)),
                "planner_left_receive_primary_fraction": float(getattr(self.base_env, "planner_left_receive_primary_fraction", 0.45)),
                "planner_left_handoff_lift_height": float(getattr(self.base_env, "planner_left_handoff_lift_height", 0.020)),
                "planner_left_pre_receive_distance": float(getattr(self.base_env, "planner_left_pre_receive_distance", 0.100)),
                "planner_left_calibrate_z_offset": float(getattr(self.base_env, "planner_left_calibrate_z_offset", 0.030)),
                "planner_left_pre_insert_height": float(getattr(self.base_env, "planner_left_pre_insert_height", 0.120)),
                "planner_left_post_release_lift_height": float(getattr(self.base_env, "planner_left_post_release_lift_height", 0.0)),
                "planner_left_receive_info": getattr(self, "last_left_receive_info", None),
                "planner_left_receive_attempt_infos": getattr(self, "left_receive_attempt_infos", []),
                "planner_left_receive_retry_count": int(getattr(self.base_env, "planner_left_receive_retry_count", 1)),
                "planner_handoff_confirm_infos": self.handoff_confirm_infos,
                "planner_state_close_infos": self.state_close_infos,
                "planner_local_cartesian_infos": self.left.local_cartesian_infos + self.right.local_cartesian_infos,
                "planner_handoff_center_xy": list(getattr(self.base_env, "planner_handoff_center_xy", ())),
                "planner_handoff_center_infos": self.handoff_center_infos,
                "planner_object_align_infos": self.object_align_infos,
                "planner_insert_readiness_infos": self.insert_readiness_infos,
                "planner_stage_snapshots": self.stage_snapshots,
                "planner_left_object_align_info": self.last_left_object_align_info,
                "planner_left_insert_readiness_info": self.last_left_insert_readiness_info,
                "planner_insert_orientation_tolerance_deg": float(getattr(self.base_env, "planner_insert_orientation_tolerance_deg", 5.0)),
                "planner_receive_z_offset_frame": "phone_local_z",
                "return_home": self.return_home,
            },
        )

    def pick_and_place(
        self,
        obj_pos: np.ndarray,
        goal_pos: np.ndarray,
        close_steps: int = 24,
        open_steps: int = 16,
    ) -> PlanningResult:
        if getattr(self.base_env, "planner_two_panda_mode", "support") == "handoff":
            return self._pick_flip_handoff_insert(obj_pos, goal_pos, close_steps, open_steps)

        completed: list[str] = []
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        pre_grasp, grasp, lift_flat, tcp_q, insert_q = self._right_phone_grasp_poses(obj_pos)

        if not self.right._run_stage("right_pre_grasp", pre_grasp, completed, self.right._refine(int(getattr(self.base_env, "planner_pre_grasp_refine_steps", 2)))):
            return PlanningResult(False, "right_pre_grasp", completed)
        if not self.right._run_local_cartesian_stage("right_grasp", grasp, completed, check_object_static=True):
            return PlanningResult(False, "right_grasp", completed, info=self._local_cartesian_info())

        self._close_gripper_until_grasp(
            "right",
            self.right,
            self.right_agent,
            completed,
            "right_close_gripper",
            self.right._refine(close_steps),
        )

        if not self.right._run_local_cartesian_stage("right_lift_flat", lift_flat, completed):
            return PlanningResult(False, "right_lift_flat", completed, info=self._local_cartesian_info())

        left_pre_support, left_support = self._left_support_poses()
        for stage, pose, refine in [
            ("left_pre_support", left_pre_support, self.left._refine(3)),
            ("left_support", left_support, self.left._refine(8)),
        ]:
            if not self.left._run_stage(stage, pose, completed, refine):
                return PlanningResult(False, stage, completed)

        failed = self._run_right_rotation_and_insert(completed, goal_pos, tcp_q, insert_q)
        if failed is not None:
            return failed

        self.right.solver.open_gripper(t=self.right._refine(open_steps))
        completed.append("right_open_gripper")

        if self.return_home:
            if not self.left.move_to_qpos(self.left_initial_qpos, refine_steps=self.left._refine(4)):
                return PlanningResult(False, "left_return_home", completed)
            completed.append("left_return_home")
            if not self.right.move_to_qpos(self.right_initial_qpos, refine_steps=self.right._refine(4)):
                return PlanningResult(False, "right_return_home", completed)
            completed.append("right_return_home")

        return PlanningResult(
            True,
            None,
            completed,
            info={
                "grasp_candidate_index": 0,
                "grasp_candidate_label": "two_panda_right_pick_left_support",
                "left_agent_uid": self.left_uid,
                "right_agent_uid": self.right_uid,
                "left_support_y_offset": float(getattr(self.base_env, "planner_left_support_y_offset", -0.120)),
                "left_support_z_offset": float(getattr(self.base_env, "planner_left_support_z_offset", 0.080)),
                "planner_insert_angle_deg": float(getattr(self.base_env, "planner_insert_angle_deg", 90.0)),
                "planner_rotation_alphas": list(getattr(self.base_env, "planner_rotation_alphas", [])),
                "planner_state_close_infos": self.state_close_infos,
                "planner_local_cartesian_infos": self.left.local_cartesian_infos + self.right.local_cartesian_infos,
                "return_home": self.return_home,
            },
        )

    def close(self) -> None:
        self.left.close()
        self.right.close()
