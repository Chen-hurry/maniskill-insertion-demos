"""Evaluation metrics for recovery experiments."""


def success_rate(successes: list[bool]) -> float:
    return sum(bool(x) for x in successes) / max(len(successes), 1)


def recovery_rate(recovered: list[bool], anomalous: list[bool]) -> float:
    total = sum(bool(x) for x in anomalous)
    if total == 0:
        return 0.0
    return sum(bool(r) and bool(a) for r, a in zip(recovered, anomalous, strict=False)) / total
