"""Environment wrappers for recovery experiments."""


class EpisodeMonitor:
    """Track episode reward, length, and success info."""

    def __init__(self, env):
        self.env = env
        self.episode_reward = 0.0
        self.episode_length = 0

    def reset(self, *args, **kwargs):
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += float(reward)
        self.episode_length += 1
        if terminated or truncated:
            info = dict(info)
            info["episode"] = {
                "reward": self.episode_reward,
                "length": self.episode_length,
                "success": bool(info.get("success", False)),
            }
        return obs, reward, terminated, truncated, info
