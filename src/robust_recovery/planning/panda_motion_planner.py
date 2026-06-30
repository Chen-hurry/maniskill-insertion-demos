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
    ) -> None:
        self.env = env
        self.prefer_screw = prefer_screw
        self.grasp_diversity = grasp_diversity
        self.grasp_candidate_count = max(1, grasp_candidate_count)
        self.refine_scale = max(1, refine_scale)
        self.rotate_on_approach = rotate_on_approach
        self.rng = np.random.default_rng(rng_seed)
        self.last_candidate_index = 0
        self.last_candidate_label = "default"
        self.last_closing_direction: list[float] | None = None
        base_env = env.unwrapped
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

    def _refine(self, steps: int) -> int:
        return steps * self.refine_scale

    def _run_stage(
        self,
        stage: str,
        pose: sapien.Pose,
        completed_stages: list[str],
        refine_steps: int = 0,
    ) -> bool:
        ok = self.move(pose, refine_steps=refine_steps)
        if ok:
            completed_stages.append(stage)
        return ok

    def _build_pickcube_waypoint_candidates(
        self,
        goal_pos: np.ndarray,
        lift_height: float = 0.12,
        pre_place_height: float = 0.10,
        place_height: float = 0.035,
    ) -> list[tuple[str, np.ndarray, PickPlaceWaypoints]]:
        env = self.env.unwrapped
        if not all(hasattr(env, name) for name in ("agent", "cube")):
            return []

        if not self.rotate_on_approach:
            tcp_pose = env.agent.tcp.pose.sp
            cube_pos = np.asarray(env.cube.pose.sp.p, dtype=np.float32)
            goal_pos = np.asarray(goal_pos, dtype=np.float32)
            tcp_q = np.asarray(tcp_pose.q, dtype=np.float32)
            grasp_pose = sapien.Pose(cube_pos, tcp_q)
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
                        lift=sapien.Pose(cube_pos + z_offset(lift_height), tcp_q),
                        pre_place=sapien.Pose(goal_pos + z_offset(pre_place_height), tcp_q),
                        place=sapien.Pose(goal_pos + z_offset(place_height), tcp_q),
                    ),
                )
            ]

        obb = get_actor_obb(env.cube)
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
                env.cube.pose.sp.p,
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
        return build_pick_place_waypoints(obj_pos, goal_pos)

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

        return PlanningResult(
            True,
            None,
            completed,
            info={
                "grasp_candidate_index": self.last_candidate_index,
                "grasp_candidate_label": self.last_candidate_label,
                "closing_direction": self.last_closing_direction,
            },
        )

    def close(self) -> None:
        self.solver.close()
