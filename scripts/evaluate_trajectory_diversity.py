#!/usr/bin/env python
"""Evaluate action-trajectory diversity for saved ManiSkill episodes."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in data.files:
        return {}
    value = data["metadata"]
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {}


def success_from_metadata(metadata: dict[str, Any]) -> bool | None:
    value = metadata.get("env_success")
    if value is None:
        return None
    return bool(value)


def episode_paths_from_report(dataset_dir: Path, successful_only: bool) -> list[Path]:
    for report_name in ("success_report.json", "summary.json"):
        report_path = dataset_dir / report_name
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        episodes = report if isinstance(report, list) else report.get("episodes", [])
        paths: list[Path] = []
        for item in episodes:
            if successful_only and not bool(item.get("env_success")):
                continue
            data_path = item.get("data_path")
            if not data_path:
                continue
            path = Path(data_path)
            if not path.is_absolute() and not path.exists():
                candidate = dataset_dir / path
                if candidate.exists():
                    path = candidate
            if path.exists():
                paths.append(path)
        if paths:
            return sorted(dict.fromkeys(paths))
    return []


def episode_paths(dataset_dir: Path, successful_only: bool) -> list[Path]:
    paths = episode_paths_from_report(dataset_dir, successful_only)
    if paths:
        return paths
    paths = sorted((dataset_dir / "episodes").glob("episode_*.npz"))
    if not successful_only:
        return paths
    successful_paths = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            metadata = load_metadata(data)
        if success_from_metadata(metadata) is not False:
            successful_paths.append(path)
    return successful_paths


def agent_label_map(metadata: dict[str, Any], action_keys: list[str]) -> dict[str, str]:
    planner_info = metadata.get("planner_result", {}).get("info", {})
    left_uid = planner_info.get("left_agent_uid")
    right_uid = planner_info.get("right_agent_uid")
    labels = {}
    for key in action_keys:
        if key == left_uid:
            labels[key] = "left"
        elif key == right_uid:
            labels[key] = "right"
        else:
            labels[key] = key
    return labels


def extract_action_trajectories(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        actions = data["actions"]
        metadata = load_metadata(data)

    if len(actions) == 0:
        raise ValueError(f"No actions in {path}")
    first = actions[0]
    if isinstance(first, dict):
        keys = sorted(str(key) for key in first.keys())
        label_map = agent_label_map(metadata, keys)
        trajectories: dict[str, list[np.ndarray]] = {label_map[key]: [] for key in keys}
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError(f"Mixed action format in {path}")
            for key in keys:
                label = label_map[key]
                trajectories[label].append(np.asarray(action[key], dtype=np.float32).reshape(-1))
        arrays = {label: np.vstack(values) for label, values in trajectories.items()}
    else:
        arrays = {"agent": np.vstack([np.asarray(action, dtype=np.float32).reshape(-1) for action in actions])}

    return {
        "path": str(path),
        "metadata": metadata,
        "success": success_from_metadata(metadata),
        "num_steps": int(len(actions)),
        "trajectories": arrays,
    }


def resample_trajectory(traj: np.ndarray, num_samples: int) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.shape[0] == num_samples:
        return traj
    if traj.shape[0] == 1:
        return np.repeat(traj, num_samples, axis=0)
    source_t = np.linspace(0.0, 1.0, traj.shape[0])
    target_t = np.linspace(0.0, 1.0, num_samples)
    columns = [np.interp(target_t, source_t, traj[:, dim]) for dim in range(traj.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def trajectory_distance(a: dict[str, Any], b: dict[str, Any], num_samples: int, gripper_weight: float) -> dict[str, Any]:
    common_agents = sorted(set(a["trajectories"]).intersection(b["trajectories"]))
    if not common_agents:
        raise ValueError(f"No common agents between {a['path']} and {b['path']}")

    per_agent: dict[str, dict[str, float]] = {}
    joint_distances: list[float] = []
    total_components: list[float] = []
    for agent in common_agents:
        ta = resample_trajectory(a["trajectories"][agent], num_samples)
        tb = resample_trajectory(b["trajectories"][agent], num_samples)
        dim = min(ta.shape[1], tb.shape[1])
        ta = ta[:, :dim]
        tb = tb[:, :dim]
        joint_dim = min(7, dim)
        joint_delta = ta[:, :joint_dim] - tb[:, :joint_dim]
        joint_l2_mean = float(np.linalg.norm(joint_delta, axis=1).mean())
        joint_l2_max = float(np.linalg.norm(joint_delta, axis=1).max())
        if dim > 7:
            gripper_abs_mean = float(np.abs(ta[:, 7] - tb[:, 7]).mean())
        else:
            gripper_abs_mean = 0.0
        combined = joint_l2_mean + gripper_weight * gripper_abs_mean
        per_agent[agent] = {
            "joint_l2_mean": joint_l2_mean,
            "joint_l2_max": joint_l2_max,
            "gripper_abs_mean": gripper_abs_mean,
            "combined": float(combined),
        }
        joint_distances.append(joint_l2_mean)
        total_components.append(combined)

    return {
        "agents": common_agents,
        "joint_l2_mean": float(np.mean(joint_distances)),
        "combined_mean": float(np.mean(total_components)),
        "per_agent": per_agent,
    }


def summarize_distances(distances: list[dict[str, Any]]) -> dict[str, Any]:
    if not distances:
        return {"num_pairs": 0}
    agents = sorted({agent for item in distances for agent in item["per_agent"]})
    per_agent = {}
    for agent in agents:
        values = [item["per_agent"][agent] for item in distances if agent in item["per_agent"]]
        per_agent[agent] = {
            "joint_l2_mean": float(np.mean([v["joint_l2_mean"] for v in values])),
            "joint_l2_max_mean": float(np.mean([v["joint_l2_max"] for v in values])),
            "gripper_abs_mean": float(np.mean([v["gripper_abs_mean"] for v in values])),
            "combined": float(np.mean([v["combined"] for v in values])),
        }
    return {
        "num_pairs": len(distances),
        "joint_l2_mean": float(np.mean([item["joint_l2_mean"] for item in distances])),
        "combined_mean": float(np.mean([item["combined_mean"] for item in distances])),
        "per_agent": per_agent,
    }


def distance_to_similarity(distance: float, tau: float) -> float:
    return float(np.exp(-float(distance) / float(tau)))


def similarity_summary(distances: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    if not distances:
        return {"num_pairs": 0}
    joint_values = [distance_to_similarity(item["joint_l2_mean"], tau) for item in distances]
    combined_values = [distance_to_similarity(item["combined_mean"], tau) for item in distances]
    agents = sorted({agent for item in distances for agent in item["per_agent"]})
    per_agent = {}
    for agent in agents:
        values = [item["per_agent"][agent] for item in distances if agent in item["per_agent"]]
        joint_agent_values = [distance_to_similarity(v["joint_l2_mean"], tau) for v in values]
        combined_agent_values = [distance_to_similarity(v["combined"], tau) for v in values]
        per_agent[agent] = {
            "joint_similarity_mean": float(np.mean(joint_agent_values)),
            "joint_similarity_min": float(np.min(joint_agent_values)),
            "joint_similarity_max": float(np.max(joint_agent_values)),
            "joint_similarity_percent_mean": float(np.mean(joint_agent_values) * 100.0),
            "combined_similarity_mean": float(np.mean(combined_agent_values)),
            "combined_similarity_percent_mean": float(np.mean(combined_agent_values) * 100.0),
        }
    return {
        "num_pairs": len(distances),
        "joint_similarity_mean": float(np.mean(joint_values)),
        "joint_similarity_min": float(np.min(joint_values)),
        "joint_similarity_max": float(np.max(joint_values)),
        "joint_similarity_percent_mean": float(np.mean(joint_values) * 100.0),
        "combined_similarity_mean": float(np.mean(combined_values)),
        "combined_similarity_percent_mean": float(np.mean(combined_values) * 100.0),
        "per_agent": per_agent,
    }


def metadata_value(metadata: dict[str, Any], dotted_key: str) -> Any:
    value: Any = metadata
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def group_value_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def grouped_similarity_report(
    episodes: list[dict[str, Any]],
    group_key: str,
    num_samples: int,
    gripper_weight: float,
    tau: float,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for index, episode in enumerate(episodes):
        raw_value = metadata_value(episode["metadata"], group_key)
        label = group_value_label(raw_value)
        groups.setdefault(label, {"value": raw_value, "episode_indices": [], "episodes": []})
        groups[label]["episode_indices"].append(index)
        groups[label]["episodes"].append(episode)

    group_reports: list[dict[str, Any]] = []
    total_pairs = 0
    weighted_joint_similarity = 0.0
    weighted_combined_similarity = 0.0
    weighted_joint_distance = 0.0
    weighted_combined_distance = 0.0

    for label, group in sorted(groups.items(), key=lambda item: item[0]):
        group_episodes = group["episodes"]
        distances = [
            trajectory_distance(a, b, num_samples, gripper_weight)
            for a, b in combinations(group_episodes, 2)
        ]
        distance_summary = summarize_distances(distances)
        sim_summary = similarity_summary(distances, tau)
        num_pairs = int(distance_summary.get("num_pairs", 0))
        joint_similarity_mean = float(sim_summary.get("joint_similarity_mean", 1.0 if len(group_episodes) == 1 else 0.0))
        combined_similarity_mean = float(sim_summary.get("combined_similarity_mean", 1.0 if len(group_episodes) == 1 else 0.0))
        joint_distance_mean = float(distance_summary.get("joint_l2_mean", 0.0))
        combined_distance_mean = float(distance_summary.get("combined_mean", 0.0))

        if num_pairs > 0:
            total_pairs += num_pairs
            weighted_joint_similarity += joint_similarity_mean * num_pairs
            weighted_combined_similarity += combined_similarity_mean * num_pairs
            weighted_joint_distance += joint_distance_mean * num_pairs
            weighted_combined_distance += combined_distance_mean * num_pairs

        group_reports.append(
            {
                "group": label,
                "value": group["value"],
                "episode_indices": group["episode_indices"],
                "num_episodes": len(group_episodes),
                "num_pairs": num_pairs,
                "pairwise_within_group": distance_summary,
                "similarity_within_group": sim_summary,
                "joint_diversity_mean": 1.0 - joint_similarity_mean,
                "joint_diversity_percent_mean": (1.0 - joint_similarity_mean) * 100.0,
                "combined_diversity_mean": 1.0 - combined_similarity_mean,
                "combined_diversity_percent_mean": (1.0 - combined_similarity_mean) * 100.0,
            }
        )

    if total_pairs > 0:
        weighted_joint_similarity /= total_pairs
        weighted_combined_similarity /= total_pairs
        weighted_joint_distance /= total_pairs
        weighted_combined_distance /= total_pairs
    else:
        weighted_joint_similarity = 1.0 if episodes else 0.0
        weighted_combined_similarity = weighted_joint_similarity

    comparable_groups = [item for item in group_reports if item["num_pairs"] > 0]
    return {
        "key": group_key,
        "num_groups": len(group_reports),
        "num_comparable_groups": len(comparable_groups),
        "total_pairs_within_groups": total_pairs,
        "weighted_summary": {
            "joint_l2_mean": weighted_joint_distance,
            "combined_mean": weighted_combined_distance,
            "joint_similarity_mean": weighted_joint_similarity,
            "joint_similarity_percent_mean": weighted_joint_similarity * 100.0,
            "joint_diversity_mean": 1.0 - weighted_joint_similarity,
            "joint_diversity_percent_mean": (1.0 - weighted_joint_similarity) * 100.0,
            "combined_similarity_mean": weighted_combined_similarity,
            "combined_similarity_percent_mean": weighted_combined_similarity * 100.0,
            "combined_diversity_mean": 1.0 - weighted_combined_similarity,
            "combined_diversity_percent_mean": (1.0 - weighted_combined_similarity) * 100.0,
        },
        "groups": group_reports,
    }


def zero_distance(episode: dict[str, Any]) -> dict[str, Any]:
    agents = sorted(episode["trajectories"])
    per_agent = {
        agent: {
            "joint_l2_mean": 0.0,
            "joint_l2_max": 0.0,
            "gripper_abs_mean": 0.0,
            "combined": 0.0,
        }
        for agent in agents
    }
    return {
        "agents": agents,
        "joint_l2_mean": 0.0,
        "combined_mean": 0.0,
        "per_agent": per_agent,
    }


def comparison_record(
    source_index: int,
    target_index: int,
    source: dict[str, Any],
    target: dict[str, Any],
    distance: dict[str, Any],
    tau: float,
) -> dict[str, Any]:
    joint_similarity = distance_to_similarity(distance["joint_l2_mean"], tau)
    combined_similarity = distance_to_similarity(distance["combined_mean"], tau)
    return {
        "target_index": target_index,
        "target_path": target["path"],
        "is_self": source_index == target_index,
        "joint_l2_mean": distance["joint_l2_mean"],
        "combined_mean": distance["combined_mean"],
        "joint_similarity": joint_similarity,
        "joint_similarity_percent": joint_similarity * 100.0,
        "combined_similarity": combined_similarity,
        "combined_similarity_percent": combined_similarity * 100.0,
    }


def trajectory_similarity_report(episodes: list[dict[str, Any]], num_samples: int, gripper_weight: float, tau: float) -> dict[str, Any]:
    distances_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    rows: list[list[float]] = []
    distance_rows: list[list[float]] = []
    per_episode: list[dict[str, Any]] = []

    for source_index, source in enumerate(episodes):
        comparisons_for_episode: list[dict[str, Any]] = []
        similarity_row: list[float] = []
        distance_row: list[float] = []
        for target_index, target in enumerate(episodes):
            if source_index == target_index:
                distance = zero_distance(source)
            else:
                key = tuple(sorted((source_index, target_index)))
                if key not in distances_by_pair:
                    distances_by_pair[key] = trajectory_distance(source, target, num_samples, gripper_weight)
                distance = distances_by_pair[key]
            record = comparison_record(source_index, target_index, source, target, distance, tau)
            comparisons_for_episode.append(record)
            similarity_row.append(record["joint_similarity"])
            distance_row.append(record["joint_l2_mean"])

        others = [item for item in comparisons_for_episode if not item["is_self"]]
        if others:
            most_similar = max(others, key=lambda item: item["joint_similarity"])
            least_similar = min(others, key=lambda item: item["joint_similarity"])
            mean_similarity = float(np.mean([item["joint_similarity"] for item in others]))
            mean_distance = float(np.mean([item["joint_l2_mean"] for item in others]))
        else:
            most_similar = None
            least_similar = None
            mean_similarity = 1.0
            mean_distance = 0.0

        per_episode.append(
            {
                "episode_index": source_index,
                "path": source["path"],
                "seed": source["metadata"].get("seed"),
                "self_similarity": comparisons_for_episode[source_index]["joint_similarity"],
                "self_distance": comparisons_for_episode[source_index]["joint_l2_mean"],
                "mean_similarity_to_others": mean_similarity,
                "mean_similarity_percent_to_others": mean_similarity * 100.0,
                "mean_distance_to_others": mean_distance,
                "most_similar": deepcopy(most_similar),
                "least_similar": deepcopy(least_similar),
                "comparisons": comparisons_for_episode,
            }
        )
        rows.append(similarity_row)
        distance_rows.append(distance_row)

    self_similarities = [row[index] for index, row in enumerate(rows)]
    self_distances = [row[index] for index, row in enumerate(distance_rows)]
    non_self_distances = [distance for pair, distance in sorted(distances_by_pair.items())]
    return {
        "tau": tau,
        "formula": "similarity = exp(-distance / tau)",
        "distance_key": "joint_l2_mean",
        "self_check": {
            "max_self_distance": float(np.max(self_distances)) if self_distances else 0.0,
            "min_self_similarity": float(np.min(self_similarities)) if self_similarities else 1.0,
            "all_self_similarity_is_one": bool(np.allclose(self_similarities, 1.0)),
        },
        "pairwise_summary_excluding_self": similarity_summary(non_self_distances, tau),
        "joint_similarity_matrix": rows,
        "joint_distance_matrix": distance_rows,
        "episodes": per_episode,
    }


def reference_similarity_report(
    episodes: list[dict[str, Any]],
    references: list[dict[str, Any]],
    distances: list[dict[str, Any]],
    tau: float,
) -> dict[str, Any]:
    comparisons_by_episode: list[dict[str, Any]] = []
    cursor = 0
    for episode_index, episode in enumerate(episodes):
        comparisons = []
        for reference_index, reference in enumerate(references):
            distance = distances[cursor]
            cursor += 1
            joint_similarity = distance_to_similarity(distance["joint_l2_mean"], tau)
            combined_similarity = distance_to_similarity(distance["combined_mean"], tau)
            comparisons.append(
                {
                    "reference_index": reference_index,
                    "reference_path": reference["path"],
                    "joint_l2_mean": distance["joint_l2_mean"],
                    "combined_mean": distance["combined_mean"],
                    "joint_similarity": joint_similarity,
                    "joint_similarity_percent": joint_similarity * 100.0,
                    "combined_similarity": combined_similarity,
                    "combined_similarity_percent": combined_similarity * 100.0,
                }
            )
        best = max(comparisons, key=lambda item: item["joint_similarity"]) if comparisons else None
        worst = min(comparisons, key=lambda item: item["joint_similarity"]) if comparisons else None
        comparisons_by_episode.append(
            {
                "episode_index": episode_index,
                "path": episode["path"],
                "seed": episode["metadata"].get("seed"),
                "mean_similarity_to_reference": float(np.mean([item["joint_similarity"] for item in comparisons])) if comparisons else 0.0,
                "mean_similarity_percent_to_reference": float(np.mean([item["joint_similarity"] for item in comparisons]) * 100.0) if comparisons else 0.0,
                "most_similar_reference": best,
                "least_similar_reference": worst,
                "comparisons": comparisons,
            }
        )
    return {
        "tau": tau,
        "formula": "similarity = exp(-distance / tau)",
        "distance_key": "joint_l2_mean",
        "summary": similarity_summary(distances, tau),
        "episodes": comparisons_by_episode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--gripper-weight", type=float, default=0.25)
    parser.add_argument("--similarity-tau", type=float, default=0.01, help="Scale for similarity = exp(-distance / tau).")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument(
        "--group-by",
        help=(
            "Dotted metadata key used for within-group diversity, "
            "e.g. initial_state_config.slot_id."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.similarity_tau <= 0.0:
        raise ValueError("--similarity-tau must be positive.")
    successful_only = not args.include_failures
    paths = episode_paths(args.dataset_dir, successful_only=successful_only)
    if not paths:
        raise SystemExit(f"No episode npz files found in {args.dataset_dir}")
    episodes = [extract_action_trajectories(path) for path in paths]

    pairwise = [
        trajectory_distance(a, b, args.num_samples, args.gripper_weight)
        for a, b in combinations(episodes, 2)
    ]
    result: dict[str, Any] = {
        "dataset_dir": str(args.dataset_dir),
        "successful_only": successful_only,
        "num_episodes": len(episodes),
        "num_samples": args.num_samples,
        "gripper_weight": args.gripper_weight,
        "episodes": [
            {
                "path": item["path"],
                "success": item["success"],
                "num_steps": item["num_steps"],
                "seed": item["metadata"].get("seed"),
                "attempt_id": item["metadata"].get("attempt_id"),
            }
            for item in episodes
        ],
        "pairwise_within_dataset": summarize_distances(pairwise),
        "trajectory_similarity": trajectory_similarity_report(
            episodes,
            args.num_samples,
            args.gripper_weight,
            args.similarity_tau,
        ),
    }
    if args.group_by:
        result["grouped_similarity"] = grouped_similarity_report(
            episodes,
            args.group_by,
            args.num_samples,
            args.gripper_weight,
            args.similarity_tau,
        )

    if args.reference_dir is not None:
        ref_paths = episode_paths(args.reference_dir, successful_only=successful_only)
        if not ref_paths:
            raise SystemExit(f"No reference episode npz files found in {args.reference_dir}")
        references = [extract_action_trajectories(path) for path in ref_paths]
        cross = [
            trajectory_distance(cur, ref, args.num_samples, args.gripper_weight)
            for cur in episodes
            for ref in references
        ]
        result["reference_dir"] = str(args.reference_dir)
        result["num_reference_episodes"] = len(references)
        result["distance_to_reference"] = summarize_distances(cross)
        result["similarity_to_reference"] = reference_similarity_report(episodes, references, cross, args.similarity_tau)
    elif len(episodes) >= 2:
        reference = episodes[0]
        to_first = [trajectory_distance(item, reference, args.num_samples, args.gripper_weight) for item in episodes[1:]]
        result["reference_episode"] = reference["path"]
        result["distance_to_first_episode"] = summarize_distances(to_first)

    output = args.output or (args.dataset_dir / "trajectory_diversity.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"episodes: {result['num_episodes']}")
    within = result["pairwise_within_dataset"]
    print(f"pairwise joint_l2_mean: {within.get('joint_l2_mean', 0.0):.6f}")
    print(f"pairwise combined_mean: {within.get('combined_mean', 0.0):.6f}")
    sim = result["trajectory_similarity"]["pairwise_summary_excluding_self"]
    print(f"pairwise joint_similarity_mean: {sim.get('joint_similarity_mean', 0.0):.6f}")
    print(f"self similarity check: {result['trajectory_similarity']['self_check']}")
    if "distance_to_reference" in result:
        ref = result["distance_to_reference"]
        print(f"to reference joint_l2_mean: {ref.get('joint_l2_mean', 0.0):.6f}")
        print(f"to reference combined_mean: {ref.get('combined_mean', 0.0):.6f}")
        ref_sim = result.get("similarity_to_reference", {}).get("summary", {})
        print(f"to reference joint_similarity_mean: {ref_sim.get('joint_similarity_mean', 0.0):.6f}")
    if "grouped_similarity" in result:
        grouped = result["grouped_similarity"]
        weighted = grouped["weighted_summary"]
        print(f"grouped by: {grouped['key']}")
        print(f"within-group pairs: {grouped['total_pairs_within_groups']}")
        print(f"within-group joint_similarity_mean: {weighted.get('joint_similarity_mean', 0.0):.6f}")
        print(f"within-group joint_diversity_mean: {weighted.get('joint_diversity_mean', 0.0):.6f}")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
