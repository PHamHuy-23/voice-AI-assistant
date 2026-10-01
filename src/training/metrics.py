import torch
import numpy as np
from typing import Tuple, List


def compute_episode_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes top-1 classification accuracy for a single episode.
    Returns accuracy as a float percentage [0.0, 100.0].
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return (correct / total) * 100.0 if total > 0 else 0.0


def compute_mean_and_confidence_interval(accuracies: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes the sample mean and 95% confidence interval for few-shot benchmark evaluation.
    CI_95 = 1.96 * (std / sqrt(num_episodes))
    """
    if not accuracies:
        return 0.0, 0.0
    arr = np.array(accuracies, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, 0.0
    std = float(np.std(arr, ddof=1))
    ci = 1.96 * (std / np.sqrt(len(arr)))
    return round(mean, 2), round(ci, 2)
