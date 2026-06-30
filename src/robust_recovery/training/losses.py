"""Loss helpers for recovery training."""


def temporal_difference_loss(predicted, target):
    return (predicted - target).pow(2).mean()


def binary_detection_loss(logits, labels):
    import torch.nn.functional as F

    return F.binary_cross_entropy_with_logits(logits, labels.float())
