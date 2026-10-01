import torch
import torch.nn.functional as F


def compute_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes pair-wise Euclidean distance squared ||x_i - y_j||^2.
    Args:
        x: Query embeddings of shape (N_query, D)
        y: Class prototypes of shape (N_class, D)
    Returns:
        Distance matrix of shape (N_query, N_class)
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1), f"Dimension mismatch: x is {d}D, y is {y.size(1)}D"

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def compute_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes pair-wise cosine distance 1 - CosineSimilarity(x_i, y_j).
    Args:
        x: (N_query, D)
        y: (N_class, D)
    Returns:
        Distance matrix of shape (N_query, N_class)
    """
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    cos_sim = torch.mm(x_norm, y_norm.t())
    return 1.0 - cos_sim
