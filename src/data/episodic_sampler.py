import random
from typing import Dict, List, Tuple, Any
from .datasets import FewShotAudioDataset


class EpisodicBatchSampler:
    """
    Episodic Sampler for N-way K-shot Few-Shot Learning.
    Guarantees:
    1. Zero overlap between Support Set and Query Set within any episode.
    2. Dynamic label remapping to contiguous integers [0, N - 1] for CrossEntropyLoss.
    3. Strict reproducibility via controlled random seeding.
    """

    def __init__(
        self,
        dataset: FewShotAudioDataset,
        n_way: int,
        k_shot: int,
        n_query: int,
        episodes_per_epoch: int,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        self.rng = random.Random(seed)

        # Check eligibility
        min_required = k_shot + n_query
        self.available_classes = self.dataset.get_valid_classes(min_samples=min_required)
        if len(self.available_classes) < n_way:
            raise ValueError(
                f"Insufficient classes with >= {min_required} samples for {n_way}-way task. "
                f"Available classes: {len(self.available_classes)}, Required: {n_way}"
            )

    def sample_episode(self) -> Dict[str, Any]:
        """
        Samples a single episode:
        - Selects N classes randomly.
        - Selects K support samples and Q query samples per class (strictly disjoint).
        - Returns structured paths and remapped labels.
        """
        selected_classes = self.rng.sample(self.available_classes, self.n_way)

        support_files: List[str] = []
        support_labels: List[int] = []
        support_class_names: List[str] = []

        query_files: List[str] = []
        query_labels: List[int] = []
        query_class_names: List[str] = []

        class_to_local_label = {cls_name: idx for idx, cls_name in enumerate(selected_classes)}

        for local_idx, cls_name in enumerate(selected_classes):
            files = self.dataset.class_to_files[cls_name]
            total_needed = self.k_shot + self.n_query
            sampled = self.rng.sample(files, total_needed)

            support_sample = sampled[: self.k_shot]
            query_sample = sampled[self.k_shot :]

            # Assert disjointness
            assert len(set(support_sample).intersection(set(query_sample))) == 0, (
                "Critical invariant violation: Overlap detected between support and query sets!"
            )

            support_files.extend(support_sample)
            support_labels.extend([local_idx] * self.k_shot)
            support_class_names.extend([cls_name] * self.k_shot)

            query_files.extend(query_sample)
            query_labels.extend([local_idx] * self.n_query)
            query_class_names.extend([cls_name] * self.n_query)

        return {
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "n_query": self.n_query,
            "selected_classes": selected_classes,
            "class_to_label": class_to_local_label,
            "support_files": support_files,
            "support_labels": support_labels,
            "support_class_names": support_class_names,
            "query_files": query_files,
            "query_labels": query_labels,
            "query_class_names": query_class_names,
        }

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            yield self.sample_episode()

    def __len__(self):
        return self.episodes_per_epoch
