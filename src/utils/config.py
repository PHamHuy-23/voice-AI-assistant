import os
import random
import yaml
import numpy as np
import torch
from typing import Any, Dict


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file safely."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    """Returns the requested device if available, otherwise falls back to cpu."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
