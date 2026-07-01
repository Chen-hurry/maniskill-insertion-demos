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
        base_env = env.unwrapped
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
        if self.prefer_screw:
            result = self.solver.move_to_pose_with_screw(
                pose,
                refine_steps=refine_steps,
            )
            if result == -1:
                result = self.solver.move_to_pose_with_RRTConnect(
                    pose,
                    refine_steps=refine_steps,
                )
        else:
            result = self.solver.move_to_pose_with_RRTConnect(
                pose,
                refine_steps=refine_steps,
            )
            if result == -1:
                result = self.solver.move_to_pose_with_screw(
                    pose,
                    refine_steps=refine_steps,
                )

        if result == -1:
            result = self.solver.move_to_pose_with_RRTStar(
                pose,
                refine_steps=refine_steps,
            )

        return result != -1

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

    def _run_stage(
        self,
        stage: str,
        pose: sapien.Pose,
        completed_stages: list[str],
        refine_steps: int = 0,
        regularized: bool = False,
    ) -> bool:
        ok = (
            self.regularized_move(pose, refine_steps=refine_steps)
            if regularized
            else self.move(pose, refine_steps=refine_steps)
        )
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
    non-selected arm holds its current joint position and inferred gripper state.
    """

    def __init__(self, env, agent_uid: str, agent) -> None:
        self.env = env
        self.agent_uid = agent_uid
        self._agent = agent

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

    def _hold_action(self, agent) -> np.ndarray:
        qpos = agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
        arm_qpos = qpos[:7]
        gripper_state = self._infer_gripper_state(agent)
        if agent.control_mode == "pd_joint_pos_vel":
            return np.hstack([arm_qpos, np.zeros_like(arm_qpos), gripper_state]).astype(np.float32)
        return np.hstack([arm_qpos, gripper_state]).astype(np.float32)

    def _compose_action(self, target_action) -> dict[str, np.ndarray]:
        action_dict: dict[str, np.ndarray] = {}
        for uid, agent in self.actual_base_env.agent.agents_dict.items():
            if uid == self.agent_uid:
                action_dict[uid] = np.asarray(target_action, dtype=np.float32)
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
        pre_grasp = sapien.Pose(grasp_pos + z_offset(0.08), tcp_q)
        grasp = sapien.Pose(grasp_pos, tcp_q)
        lift_flat = sapien.Pose(grasp_pos + z_offset(0.16), tcp_q)
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

        for stage, pose, refine in [
            ("pre_grasp", pre_grasp, self._refine(2)),
            ("grasp", grasp, self._refine(10)),
        ]:
            if not self._run_stage(stage, pose, completed, refine):
                return PlanningResult(False, stage, completed)

        self.solver.close_gripper(t=self._refine(close_steps))
        completed.append("close_gripper")

        if not self._run_stage("lift_flat", lift_flat, completed, self._refine(4)):
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
        pre_insert_pos = goal_pos + tcp_minus_obj + z_offset(0.12)
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
            SingleAgentControlAdapter(env, self.right_uid, self.right_agent),
            return_home=False,
            home_qpos=home_qpos,
            **common_kwargs,
        )
        self.left = PandaPickPlacePlanner(
            SingleAgentControlAdapter(env, self.left_uid, self.left_agent),
            return_home=False,
            home_qpos=self.left_initial_qpos,
            **common_kwargs,
        )


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

        # Top-down width grasp: keep the gripper pointing down and close across
        # the phone width in the table plane. This avoids pinching the 5 mm
        # thickness edge, which is unstable during wrist flipping.
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
        pre_grasp = sapien.Pose(grasp_pos + z_offset(0.08), tcp_q)
        grasp = sapien.Pose(grasp_pos, tcp_q)
        lift_flat = sapien.Pose(grasp_pos + z_offset(0.16), tcp_q)
        return pre_grasp, grasp, lift_flat, tcp_q, insert_q

    def _current_obj_pos(self) -> np.ndarray:
        return np.asarray(self.base_env.cube.pose.sp.p, dtype=np.float32)

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
            rotate_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(0.04)
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
        pre_insert_pos = goal_pos + tcp_minus_obj + z_offset(0.12)
        insert_pos = goal_pos + tcp_minus_obj
        use_regularized_insert = getattr(self.base_env, "planner_use_regularized_insert", True)
        for stage, pose, refine in [
            ("pre_insert", sapien.Pose(pre_insert_pos, insert_q), self.right._refine(4)),
            ("insert", sapien.Pose(insert_pos, insert_q), self.right._refine(14)),
        ]:
            if not self.right._run_stage(stage, pose, completed, refine, regularized=use_regularized_insert):
                return PlanningResult(False, stage, completed)
        return None


    def _world_y_rotation_quat(self, angle_deg: float) -> np.ndarray:
        half = -np.deg2rad(float(angle_deg)) * 0.5
        return np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float32)

    def _left_receive_poses(self) -> tuple[sapien.Pose, sapien.Pose, np.ndarray]:
        env = self.base_env
        obj_pos = self._current_obj_pos()

        receive_mode = getattr(env, "planner_handoff_receive_mode", "topdown_center")
        handoff_angle = float(getattr(env, "planner_handoff_angle_deg", 45.0))
        theta = np.deg2rad(handoff_angle)
        receive_pos = obj_pos.copy()

        if receive_mode == "tilted_face":
            # Experimental: approach from the left side and close along the
            # tilted phone face normal, so the receiver grips the phone's broad
            # front/back faces instead of the width side faces held by the
            # flipping arm. This is physically harder because the phone is only
            # 5 mm thick.
            approaching = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            closing = np.array([-np.sin(theta), 0.0, np.cos(theta)], dtype=np.float32)
            closing = closing / np.linalg.norm(closing)
        else:
            # Approach from above and close across the phone width. In
            # upper_side mode the receiver grasps the raised upper section of
            # the phone rather than the center, leaving clearance for the
            # flipping gripper.
            approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            closing = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if receive_mode == "upper_side":
                phone_half_size = np.asarray(getattr(env, "phone_half_size", (0.075, 0.025, 0.0025)), dtype=np.float32)
                fraction = float(getattr(env, "planner_upper_side_receive_fraction", 0.08))
                length_axis = np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float32)
                receive_pos = receive_pos + length_axis * float(phone_half_size[0] * fraction)

        receive_pos[2] += float(getattr(env, "planner_left_receive_z_offset", 0.004))
        receive_pose = self.left_agent.build_grasp_pose(approaching, closing, receive_pos)
        pre_receive = receive_pose * sapien.Pose([0.0, 0.0, -0.100])
        return pre_receive, receive_pose, np.asarray(receive_pose.q, dtype=np.float32)

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
        if abs(delta_angle) > 1.0:
            rotate_pos = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(0.030)
            for alpha in (0.5, 1.0):
                q = quat_mul(self._world_y_rotation_quat(delta_angle * alpha), receive_q)
                stage = f"left_calibrate_{int(round(handoff_angle + delta_angle * alpha))}"
                if not self.left._run_stage(
                    stage,
                    sapien.Pose(rotate_pos, q),
                    completed,
                    self.left._refine(5),
                ):
                    return PlanningResult(False, stage, completed)
            insert_q = quat_mul(self._world_y_rotation_quat(delta_angle), receive_q)
        else:
            completed.append(f"left_calibrate_{int(round(target_angle))}_skipped")

        tcp_pos = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32)
        obj_pos = self._current_obj_pos()
        tcp_minus_obj = tcp_pos - obj_pos
        pre_insert_pos = goal_pos + tcp_minus_obj + z_offset(0.12)
        insert_pos = goal_pos + tcp_minus_obj
        use_regularized_insert = getattr(self.base_env, "planner_use_regularized_insert", True)
        for stage, pose, refine in [
            ("left_pre_insert", sapien.Pose(pre_insert_pos, insert_q), self.left._refine(4)),
            ("left_insert", sapien.Pose(insert_pos, insert_q), self.left._refine(14)),
        ]:
            if not self.left._run_stage(stage, pose, completed, refine, regularized=use_regularized_insert):
                return PlanningResult(False, stage, completed)
        return None

    def _pick_flip_handoff_insert(
        self,
        obj_pos: np.ndarray,
        goal_pos: np.ndarray,
        close_steps: int,
        open_steps: int,
    ) -> PlanningResult:
        completed: list[str] = []
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        pre_grasp, grasp, lift_flat, tcp_q, insert_q = self._right_phone_grasp_poses(obj_pos)

        for stage, pose, refine in [
            ("right_pre_grasp", pre_grasp, self.right._refine(2)),
            ("right_grasp", grasp, self.right._refine(10)),
        ]:
            if not self.right._run_stage(stage, pose, completed, refine):
                return PlanningResult(False, stage, completed)

        self.right.solver.close_gripper(t=self.right._refine(close_steps))
        completed.append("right_close_gripper")

        if not self.right._run_stage("right_lift_flat", lift_flat, completed, self.right._refine(4)):
            return PlanningResult(False, "right_lift_flat", completed)

        target_angle = float(getattr(self.base_env, "planner_insert_angle_deg", 90.0))
        handoff_angle = float(getattr(self.base_env, "planner_handoff_angle_deg", 45.0))
        alpha = float(np.clip(handoff_angle / max(abs(target_angle), 1e-6), 0.05, 1.0))
        rotate_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(0.04)
        if not self.right._run_rotation_stage(
            f"right_flip_handoff_{int(round(handoff_angle))}",
            rotate_pos,
            tcp_q,
            insert_q,
            (alpha,),
            completed,
        ):
            return PlanningResult(False, "right_flip_handoff", completed)

        left_pre_receive, left_receive, receive_q = self._left_receive_poses()
        for stage, pose, refine in [
            ("left_pre_receive", left_pre_receive, self.left._refine(4)),
            ("left_receive", left_receive, self.left._refine(10)),
        ]:
            if not self.left._run_stage(stage, pose, completed, refine):
                return PlanningResult(False, stage, completed)

        self.left.solver.close_gripper(t=self.left._refine(close_steps))
        completed.append("left_close_gripper")
        self.right.solver.open_gripper(t=self.right._refine(open_steps))
        completed.append("right_release_handoff")

        right_retract_pos = np.asarray(self.right_agent.tcp.pose.sp.p, dtype=np.float32) + np.array([0.0, 0.08, 0.08], dtype=np.float32)
        right_retract_q = np.asarray(self.right_agent.tcp.pose.sp.q, dtype=np.float32)
        if not self.right._run_stage(
            "right_retract_after_handoff",
            sapien.Pose(right_retract_pos, right_retract_q),
            completed,
            self.right._refine(4),
        ):
            return PlanningResult(False, "right_retract_after_handoff", completed)

        left_lift_pos = np.asarray(self.left_agent.tcp.pose.sp.p, dtype=np.float32) + z_offset(float(getattr(self.base_env, "planner_left_handoff_lift_height", 0.020)))
        if not self.left._run_stage(
            "left_lift_after_handoff",
            sapien.Pose(left_lift_pos, receive_q),
            completed,
            self.left._refine(6),
        ):
            return PlanningResult(False, "left_lift_after_handoff", completed)

        failed = self._run_left_calibrate_and_insert(completed, goal_pos, receive_q)
        if failed is not None:
            return failed

        self.left.solver.open_gripper(t=self.left._refine(open_steps))
        completed.append("left_open_gripper")

        if self.return_home:
            if not self.right.move_to_qpos(self.right_initial_qpos, refine_steps=self.right._refine(4)):
                return PlanningResult(False, "right_return_home", completed)
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
                "left_agent_uid": self.left_uid,
                "right_agent_uid": self.right_uid,
                "handoff_angle_deg": handoff_angle,
                "planner_insert_angle_deg": target_angle,
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

        for stage, pose, refine in [
            ("right_pre_grasp", pre_grasp, self.right._refine(2)),
            ("right_grasp", grasp, self.right._refine(10)),
        ]:
            if not self.right._run_stage(stage, pose, completed, refine):
                return PlanningResult(False, stage, completed)

        self.right.solver.close_gripper(t=self.right._refine(close_steps))
        completed.append("right_close_gripper")

        if not self.right._run_stage("right_lift_flat", lift_flat, completed, self.right._refine(4)):
            return PlanningResult(False, "right_lift_flat", completed)

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
                "return_home": self.return_home,
            },
        )

    def close(self) -> None:
        self.left.close()
        self.right.close()

