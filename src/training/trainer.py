import os
import json
import time
from typing import Dict, Any, Optional
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from .losses import PrototypicalLoss
from .metrics import compute_episode_accuracy, compute_mean_and_confidence_interval
from ..data.episodic_sampler import EpisodicBatchSampler
from ..data.preprocessing import AudioPreprocessor
from ..features.audio_features import FeatureExtractor
from ..features.visualization import plot_training_history
from ..models.prototypical_network import PrototypicalNetwork


class EpisodicTrainer:
    """
    Manages episodic training, validation, and checkpoint artifact generation.
    Purely config-driven and separated from presentation notebooks.
    """

    def __init__(
        self,
        model: PrototypicalNetwork,
        train_sampler: EpisodicBatchSampler,
        val_sampler: Optional[EpisodicBatchSampler],
        preprocessor: AudioPreprocessor,
        feature_extractor: FeatureExtractor,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.save_dir = train_cfg.get("save_checkpoint_dir", "checkpoints/default")
        os.makedirs(self.save_dir, exist_ok=True)

        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = StepLR(
            self.optimizer,
            step_size=train_cfg.get("lr_step_size", 15),
            gamma=train_cfg.get("lr_gamma", 0.5),
        )
        self.criterion = PrototypicalLoss()

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_times_sec": [],
        }
        self.best_val_acc = 0.0

    def _prepare_batch_tensors(self, file_paths: list) -> torch.Tensor:
        """Loads, preprocesses, and extracts feature tensors for a list of audio files."""
        tensors = []
        for fp in file_paths:
            audio, sr, _ = self.preprocessor.process(fp)
            feat = self.feature_extractor.extract_model_input(audio)
            tensors.append(feat)
        return torch.cat(tensors, dim=0).to(self.device)  # (B, C, F, T)

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        losses = []
        accuracies = []

        for episode in self.train_sampler:
            support_tensors = self._prepare_batch_tensors(episode["support_files"])
            query_tensors = self._prepare_batch_tensors(episode["query_files"])
            query_labels = torch.tensor(episode["query_labels"], dtype=torch.long, device=self.device)

            self.optimizer.zero_grad()
            logits, _, _ = self.model.forward_episode(
                support_inputs=support_tensors,
                query_inputs=query_tensors,
                n_way=episode["n_way"],
                k_shot=episode["k_shot"],
            )

            loss = self.criterion(logits, query_labels)
            loss.backward()
            self.optimizer.step()

            acc = compute_episode_accuracy(logits, query_labels)
            losses.append(loss.item())
            accuracies.append(acc)

        return float(torch.tensor(losses).mean()), float(torch.tensor(accuracies).mean())

    def validate_epoch(self) -> Tuple[float, float]:
        if not self.val_sampler:
            return 0.0, 0.0

        self.model.eval()
        losses = []
        accuracies = []

        with torch.no_grad():
            for episode in self.val_sampler:
                support_tensors = self._prepare_batch_tensors(episode["support_files"])
                query_tensors = self._prepare_batch_tensors(episode["query_files"])
                query_labels = torch.tensor(episode["query_labels"], dtype=torch.long, device=self.device)

                logits, _, _ = self.model.forward_episode(
                    support_inputs=support_tensors,
                    query_inputs=query_tensors,
                    n_way=episode["n_way"],
                    k_shot=episode["k_shot"],
                )

                loss = self.criterion(logits, query_labels)
                acc = compute_episode_accuracy(logits, query_labels)
                losses.append(loss.item())
                accuracies.append(acc)

        return float(torch.tensor(losses).mean()), float(torch.tensor(accuracies).mean())

    def fit(self) -> Dict[str, Any]:
        """Runs full training across all epochs and exports experiment artifacts."""
        print(f"=== Starting Episodic Training ({self.epochs} Epochs) on device: {self.device} ===")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch()
            self.scheduler.step()
            duration = round(time.time() - t0, 2)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_times_sec"].append(duration)

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {duration}s"
            )

            # Checkpoint saving
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "val_acc": val_acc,
                        "config": self.config,
                    },
                    os.path.join(self.save_dir, "best.pt"),
                )

            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "val_acc": val_acc,
                    "config": self.config,
                },
                os.path.join(self.save_dir, "last.pt"),
            )

        # Save Metrics JSON
        metrics_path = os.path.join(self.save_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val_acc": self.best_val_acc,
                    "final_train_acc": self.history["train_acc"][-1],
                    "history": self.history,
                },
                f,
                indent=2,
            )

        # Plot Curves
        curves_path = os.path.join(self.save_dir, "training_curves.png")
        plot_training_history(self.history, curves_path)
        print(f"Artifacts successfully saved to: {self.save_dir}")
        return self.history
