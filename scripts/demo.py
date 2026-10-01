import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_yaml, get_device
from src.data.preprocessing import AudioPreprocessor
from src.features.audio_features import FeatureExtractor
from src.models.td_resnet import TDResNetEmbeddingNet
from src.models.prototypical_network import PrototypicalNetwork


class FewShotKWSInferenceEngine:
    """
    Inference Engine for Few-Shot Keyword Spotting.
    Serves as the Core Audio Classifier bridge for the Desktop Voice Assistant.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_config_path: str = "configs/model/td_resnet.yaml",
        proto_config_path: str = "configs/model/prototypical.yaml",
        prep_config_path: str = "configs/preprocessing/normalized.yaml",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        model_cfg = load_yaml(model_config_path)
        proto_cfg = load_yaml(proto_config_path)
        prep_cfg = load_yaml(prep_config_path)

        self.preprocessor = AudioPreprocessor(prep_cfg)
        self.feature_extractor = FeatureExtractor(model_cfg)

        self.backbone = TDResNetEmbeddingNet(model_cfg)
        self.model = PrototypicalNetwork(self.backbone, proto_cfg).to(self.device)
        self.model.eval()

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            print(f"[+] Model checkpoint loaded successfully: {checkpoint_path}")
        else:
            print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")

        self.support_classes: List[str] = []
        self.prototypes: Optional[torch.Tensor] = None

    def register_support_set(self, support_dir: str, file_ext: str = ".wav") -> None:
        """
        Indexes keyword folders under support_dir, encodes support samples,
        and computes reference class prototypes.
        Example structure:
            support_dir/
                yes/ [sample1.wav, sample2.wav]
                stop/ [sample1.wav, sample2.wav]
        """
        subdirs = sorted([d for d in os.listdir(support_dir) if os.path.isdir(os.path.join(support_dir, d))])
        if not subdirs:
            raise ValueError(f"No keyword subdirectories found in support directory: {support_dir}")

        self.support_classes = subdirs
        prototypes_list = []

        with torch.no_grad():
            for cls_name in self.support_classes:
                cls_folder = os.path.join(support_dir, cls_name)
                files = sorted(glob.glob(os.path.join(cls_folder, f"*{file_ext}")))
                if not files:
                    raise ValueError(f"Support class '{cls_name}' has 0 audio files!")

                class_embeddings = []
                for fp in files:
                    audio, _, _ = self.preprocessor.process(fp)
                    feat = self.feature_extractor.extract_model_input(audio).to(self.device)
                    emb = self.backbone(feat)
                    class_embeddings.append(emb)

                class_embeddings = torch.cat(class_embeddings, dim=0)  # (K, D)
                cls_proto = class_embeddings.mean(dim=0, keepdim=True)  # (1, D)
                cls_proto = F.normalize(cls_proto, p=2, dim=1)
                prototypes_list.append(cls_proto)

        self.prototypes = torch.cat(prototypes_list, dim=0)  # (N, D)
        print(f"[+] Successfully registered {len(self.support_classes)} keywords into prototype memory.")

    def predict(self, query_audio_path: str) -> Dict[str, Any]:
        """
        Classifies a single query audio file against the registered keyword prototypes.
        """
        if self.prototypes is None:
            raise RuntimeError("Prototypes have not been initialized. Call register_support_set first.")

        with torch.no_grad():
            audio, sr, _ = self.preprocessor.process(query_audio_path)
            feat = self.feature_extractor.extract_model_input(audio).to(self.device)
            query_emb = self.backbone(feat)

            # Compute distances to prototypes
            predicted_idx, distances = self.model.predict_query(query_emb, self.prototypes)
            dists = distances.squeeze(0)  # (N,)

            # Softmax confidence score
            logits = -dists / self.model.temperature
            probs = F.softmax(logits, dim=0)

            best_idx = predicted_idx.item()
            predicted_keyword = self.support_classes[best_idx]
            confidence = float(probs[best_idx].item()) * 100.0

            distance_dict = {
                cls_name: round(float(dists[i].item()), 4)
                for i, cls_name in enumerate(self.support_classes)
            }

            return {
                "query_file": os.path.basename(query_audio_path),
                "predicted_keyword": predicted_keyword,
                "confidence_percent": round(confidence, 2),
                "distances": distance_dict,
            }


def main():
    parser = argparse.ArgumentParser(description="Few-Shot Keyword Spotting Demo CLI.")
    parser.add_argument("--support", type=str, default="demo/support", help="Path to support samples folder.")
    parser.add_argument("--query", type=str, required=True, help="Path to query audio WAV file.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to model checkpoint.")
    args = parser.parse_args()

    print("=======================================================")
    print("  Few-Shot Keyword Spotting (TD-ResNet + ProtoNet) Demo")
    print("=======================================================")

    engine = FewShotKWSInferenceEngine(checkpoint_path=args.checkpoint)
    engine.register_support_set(args.support)

    result = engine.predict(args.query)

    print("\n--- INFERENCE RESULT ---")
    print(f"Query Audio:       {result['query_file']}")
    print(f"Predicted Keyword: >>> {result['predicted_keyword'].upper()} <<<")
    print(f"Confidence Score:  {result['confidence_percent']}%")
    print("Class Distances (lower is closer):")
    for cls_name, dist in result["distances"].items():
        bar = "=" * int(max(1, 20 - int(dist * 10)))
        print(f"  - {cls_name:12s}: {dist:.4f}  {bar}")


if __name__ == "__main__":
    main()
