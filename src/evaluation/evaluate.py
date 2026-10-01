import time
from typing import Dict, List, Any, Optional
import torch
import numpy as np

from ..data.episodic_sampler import EpisodicBatchSampler
from ..data.preprocessing import AudioPreprocessor
from ..features.audio_features import FeatureExtractor
from ..models.prototypical_network import PrototypicalNetwork
from ..training.metrics import compute_episode_accuracy, compute_mean_and_confidence_interval


class FewShotEvaluator:
    """
    Evaluates trained Few-Shot KWS models across multiple N-way K-shot configurations.
    Measures accuracy, 95% confidence intervals, and inference latency.
    """

    def __init__(
        self,
        model: PrototypicalNetwork,
        preprocessor: AudioPreprocessor,
        feature_extractor: FeatureExtractor,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.device = device

    def _prepare_batch_tensors(self, file_paths: list) -> torch.Tensor:
        tensors = []
        for fp in file_paths:
            audio, sr, _ = self.preprocessor.process(fp)
            feat = self.feature_extractor.extract_model_input(audio)
            tensors.append(feat)
        return torch.cat(tensors, dim=0).to(self.device)

    def evaluate_sampler(self, sampler: EpisodicBatchSampler) -> Dict[str, Any]:
        """Runs evaluation over an episodic test sampler and computes metrics."""
        accuracies: List[float] = []
        latencies_ms: List[float] = []

        with torch.no_grad():
            for episode in sampler:
                support_tensors = self._prepare_batch_tensors(episode["support_files"])
                query_tensors = self._prepare_batch_tensors(episode["query_files"])
                query_labels = torch.tensor(episode["query_labels"], dtype=torch.long, device=self.device)

                t0 = time.perf_counter()
                logits, _, _ = self.model.forward_episode(
                    support_inputs=support_tensors,
                    query_inputs=query_tensors,
                    n_way=episode["n_way"],
                    k_shot=episode["k_shot"],
                )
                t1 = time.perf_counter()

                acc = compute_episode_accuracy(logits, query_labels)
                accuracies.append(acc)
                latencies_ms.append((t1 - t0) * 1000.0)

        mean_acc, ci_95 = compute_mean_and_confidence_interval(accuracies)
        mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        return {
            "n_way": sampler.n_way,
            "k_shot": sampler.k_shot,
            "num_episodes": len(sampler),
            "mean_accuracy": mean_acc,
            "ci_95": ci_95,
            "mean_latency_ms": round(mean_latency, 2),
            "raw_accuracies": accuracies,
        }
