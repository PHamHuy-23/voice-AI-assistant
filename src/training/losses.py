import torch
import torch.nn as nn


class PrototypicalLoss(nn.Module):
    """
    Computes Cross Entropy Loss over the distance-derived negative log likelihood logits.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N * Q, N)
            target_labels: (N * Q,) with values in [0, N-1]
        """
        return self.criterion(logits, target_labels)
