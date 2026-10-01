import os
from typing import List, Optional
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..models.td_resnet import TDResNetEmbeddingNet
from ..data.preprocessing import AudioPreprocessor
from ..features.audio_features import FeatureExtractor


class EmbeddingVisualizer:
    """
    Extracts embeddings from TD-ResNet and projects them onto 2D space
    via PCA or t-SNE to inspect cluster separability across spoken keywords.
    """

    def __init__(
        self,
        backbone: TDResNetEmbeddingNet,
        preprocessor: AudioPreprocessor,
        feature_extractor: FeatureExtractor,
        device: torch.device,
    ):
        self.backbone = backbone.to(device)
        self.backbone.eval()
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.device = device

    def extract_dataset_embeddings(self, file_paths: List[str], class_names: List[str]):
        embeddings = []
        labels = []
        with torch.no_grad():
            for fp, cls in zip(file_paths, class_names):
                audio, _, _ = self.preprocessor.process(fp)
                feat = self.feature_extractor.extract_model_input(audio).to(self.device)
                emb = self.backbone(feat).cpu().numpy().squeeze(0)
                embeddings.append(emb)
                labels.append(cls)
        return np.array(embeddings), labels

    def plot_projections(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        method: str = "tsne",
        title: str = "Embedding Space Clusters",
        save_path: Optional[str] = None,
    ) -> None:
        """Projects high-dimensional embeddings to 2D using PCA or t-SNE and plots clusters."""
        if method.lower() == "pca":
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(embeddings)
        else:
            # t-SNE
            perplexity = min(30, max(5, len(embeddings) // 4))
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            coords = reducer.fit_transform(embeddings)

        unique_labels = sorted(list(set(labels)))
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))

        plt.figure(figsize=(9, 7))
        for idx, lbl in enumerate(unique_labels):
            mask = [l == lbl for l in labels]
            pts = coords[mask]
            plt.scatter(pts[:, 0], pts[:, 1], label=lbl, color=cmap(idx), alpha=0.8, s=40)

        plt.title(f"{title} ({method.upper()})", fontsize=13, fontweight="bold")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
        plt.close()
