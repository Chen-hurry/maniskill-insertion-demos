"""Thin adapter around ManiSkill-style environments."""

from dataclasses import dataclass
from typing import Any


@dataclass
class EnvConfig:
    task_id: str
    observation_mode: str = "rgbd"
    control_mode: str = "pd_ee_delta_pose"
    max_episode_steps: int = 200


class ManiSkillEnv:
    """Lazy environment wrapper so the repository can import without ManiSkill installed."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.env: Any | None = None

    def build(self) -> Any:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError("Install the `maniskill` extra or gymnasium-compatible envs.") from exc

        self.env = gym.make(
            self.config.task_id,
            obs_mode=self.config.observation_mode,
            control_mode=self.config.control_mode,
            max_episode_steps=self.config.max_episode_steps,
        )
        return self.env
