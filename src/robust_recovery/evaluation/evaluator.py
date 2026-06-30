"""Policy evaluation loop scaffold."""

from robust_recovery.evaluation.metrics import success_rate


class Evaluator:
    def __init__(self, episodes: int = 50):
        self.episodes = episodes

    def evaluate(self, policy=None) -> dict[str, float]:
        successes = [False for _ in range(self.episodes)]
        return {"success_rate": success_rate(successes), "episodes": float(self.episodes)}
