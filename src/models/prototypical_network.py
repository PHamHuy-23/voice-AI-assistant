import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .distances import compute_euclidean_distance, compute_cosine_distance


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for Few-Shot Keyword Spotting.
    Wraps an embedding backbone f_theta(x) (e.g. TD-ResNet).
    Prototypes are dynamically computed per episode as:
        c_k = (1 / K) * sum_{i in S_k} f_theta(x_i)
    Logits are inversely proportional to distances:
        p(y = k | x_q) = softmax( -d(f_theta(x_q), c_k) / temperature )
    """

    def __init__(self, backbone: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.backbone = backbone
        self.distance_metric = config.get("distance_metric", "euclidean")
        self.temperature = float(config.get("temperature", 1.0))

    def compute_prototypes(self, support_embeddings: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        """
        Computes prototypes c_k for each of the N classes.
        Args:
            support_embeddings: (N * K, D)
        Returns:
            prototypes: (N, D)
        """
        d = support_embeddings.size(-1)
        # Reshape to (N, K, D) and take the mean along K
        prototypes = support_embeddings.view(n_way, k_shot, d).mean(dim=1)
        return prototypes

    def forward_episode(
        self,
        support_inputs: torch.Tensor,
        query_inputs: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes an entire episode:
        Args:
            support_inputs: (N * K, C, F, T)
            query_inputs:   (N * Q, C, F, T)
            n_way: Number of classes in this episode
            k_shot: Number of support examples per class
        Returns:
            logits: (N * Q, N)
            distances: (N * Q, N)
            prototypes: (N, D)
        """
        support_embeddings = self.backbone(support_inputs)
        query_embeddings = self.backbone(query_inputs)

        prototypes = self.compute_prototypes(support_embeddings, n_way, k_shot)

        if self.distance_metric == "cosine":
            distances = compute_cosine_distance(query_embeddings, prototypes)
        else:
            distances = compute_euclidean_distance(query_embeddings, prototypes)

        logits = -distances / self.temperature
        return logits, distances, prototypes

    def predict_query(self, query_inputs: torch.Tensor, prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference step: Given precomputed prototypes, predicts the nearest class for query inputs.
        Returns:
            predicted_class_indices: (B,)
            distances: (B, N)
        """
        query_embeddings = self.backbone(query_inputs)
        if self.distance_metric == "cosine":
            distances = compute_cosine_distance(query_embeddings, prototypes)
        else:
            distances = compute_euclidean_distance(query_embeddings, prototypes)

        predicted_indices = torch.argmin(distances, dim=1)
        return predicted_indices, distances
