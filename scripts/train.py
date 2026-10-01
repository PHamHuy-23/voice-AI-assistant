import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_yaml, set_seed, get_device
from src.data.datasets import FewShotAudioDataset
from src.data.episodic_sampler import EpisodicBatchSampler
from src.data.preprocessing import AudioPreprocessor
from src.features.audio_features import FeatureExtractor
from src.models.td_resnet import TDResNetEmbeddingNet
from src.models.prototypical_network import PrototypicalNetwork
from src.training.trainer import EpisodicTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Few-Shot KWS with TD-ResNet and Prototypical Networks.")
    parser.add_argument("--config", type=str, default="configs/experiments/exp_001_5way_5shot.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Override raw dataset folder.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    args = parser.parse_args()

    exp_cfg = load_yaml(args.config)
    seed = exp_cfg.get("random_seed", 42)
    set_seed(seed)

    data_cfg = load_yaml(exp_cfg["data_config"])
    prep_cfg = load_yaml(exp_cfg["preprocessing_config"])
    model_cfg = load_yaml(exp_cfg["model_config"])
    proto_cfg = load_yaml(exp_cfg["protonet_config"])

    if args.epochs:
        exp_cfg["training"]["epochs"] = args.epochs

    data_dir = args.data_dir or data_cfg.get("data_dir", "data/raw/speech_commands_v2")
    splits = data_cfg.get("splits", {})
    train_classes = splits.get("train_classes", [])
    val_classes = splits.get("val_classes", [])

    print(f"[*] Training on: {data_dir}")
    print(f"[*] Train Classes ({len(train_classes)}): {train_classes}")
    print(f"[*] Val Classes ({len(val_classes)}): {val_classes}")

    train_dataset = FewShotAudioDataset(data_dir=data_dir, classes=train_classes)
    val_dataset = FewShotAudioDataset(data_dir=data_dir, classes=val_classes)

    episodic_cfg = proto_cfg.get("episodic", {})
    train_sampler = EpisodicBatchSampler(
        dataset=train_dataset,
        n_way=episodic_cfg.get("n_way", 5),
        k_shot=episodic_cfg.get("k_shot", 5),
        n_query=episodic_cfg.get("n_query", 5),
        episodes_per_epoch=exp_cfg["training"].get("episodes_per_epoch", 100),
        seed=seed,
    )

    val_sampler = None
    if val_classes and len(val_dataset.get_valid_classes()) >= episodic_cfg.get("val_n_way", 5):
        val_sampler = EpisodicBatchSampler(
            dataset=val_dataset,
            n_way=episodic_cfg.get("val_n_way", 5),
            k_shot=episodic_cfg.get("val_k_shot", 5),
            n_query=episodic_cfg.get("val_n_query", 5),
            episodes_per_epoch=exp_cfg["training"].get("val_episodes", 50),
            seed=seed + 1,
        )

    preprocessor = AudioPreprocessor(prep_cfg)
    feature_extractor = FeatureExtractor(model_cfg)

    backbone = TDResNetEmbeddingNet(model_cfg)
    model = PrototypicalNetwork(backbone, proto_cfg)

    device = get_device(exp_cfg["training"].get("device", "cuda"))

    trainer = EpisodicTrainer(
        model=model,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        config=exp_cfg,
        device=device,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
