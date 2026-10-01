import argparse
import os
import sys
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_yaml, set_seed, get_device
from src.data.datasets import FewShotAudioDataset
from src.data.episodic_sampler import EpisodicBatchSampler
from src.data.preprocessing import AudioPreprocessor
from src.features.audio_features import FeatureExtractor
from src.models.td_resnet import TDResNetEmbeddingNet
from src.models.prototypical_network import PrototypicalNetwork
from src.evaluation.evaluate import FewShotEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Few-Shot KWS Checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--config", type=str, default="configs/experiments/exp_001_5way_5shot.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Override test data folder.")
    parser.add_argument("--output_csv", type=str, default="reports/tables/evaluation_results.csv")
    args = parser.parse_args()

    exp_cfg = load_yaml(args.config)
    seed = exp_cfg.get("random_seed", 42)
    set_seed(seed)

    data_cfg = load_yaml(exp_cfg["data_config"])
    prep_cfg = load_yaml(exp_cfg["preprocessing_config"])
    model_cfg = load_yaml(exp_cfg["model_config"])
    proto_cfg = load_yaml(exp_cfg["protonet_config"])

    data_dir = args.data_dir or data_cfg.get("data_dir", "data/raw/speech_commands_v2")
    test_classes = data_cfg.get("splits", {}).get("test_classes", [])

    print(f"[*] Evaluating on Test Classes ({len(test_classes)}): {test_classes}")

    test_dataset = FewShotAudioDataset(data_dir=data_dir, classes=test_classes)
    preprocessor = AudioPreprocessor(prep_cfg)
    feature_extractor = FeatureExtractor(model_cfg)

    backbone = TDResNetEmbeddingNet(model_cfg)
    model = PrototypicalNetwork(backbone, proto_cfg)

    device = get_device("cuda")
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[+] Loaded weights from: {args.checkpoint}")
    else:
        print(f"[!] Warning: Checkpoint not found at {args.checkpoint}. Running with initialized weights.")

    evaluator = FewShotEvaluator(model, preprocessor, feature_extractor, device)

    configs_to_eval = exp_cfg.get("evaluation", {}).get(
        "eval_configurations",
        [
            {"n_way": 5, "k_shot": 1, "n_query": 5},
            {"n_way": 5, "k_shot": 5, "n_query": 5},
            {"n_way": 10, "k_shot": 1, "n_query": 5},
            {"n_way": 10, "k_shot": 5, "n_query": 5},
        ],
    )

    results = []
    num_episodes = exp_cfg.get("evaluation", {}).get("test_episodes", 200)

    for cfg in configs_to_eval:
        n_way = cfg["n_way"]
        k_shot = cfg["k_shot"]
        n_query = cfg.get("n_query", 5)

        sampler = EpisodicBatchSampler(
            dataset=test_dataset,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            episodes_per_epoch=num_episodes,
            seed=seed + 100,
        )

        res = evaluator.evaluate_sampler(sampler)
        print(
            f"[*] Configuration: {n_way}-way {k_shot}-shot | "
            f"Accuracy: {res['mean_accuracy']}% ± {res['ci_95']}% | "
            f"Latency: {res['mean_latency_ms']}ms/ep"
        )
        results.append(
            {
                "N-way": n_way,
                "K-shot": k_shot,
                "Episodes": num_episodes,
                "Mean Accuracy (%)": res["mean_accuracy"],
                "95% CI (%)": res["ci_95"],
                "Mean Latency (ms)": res["mean_latency_ms"],
            }
        )

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    res_df.to_csv(args.output_csv, index=False)
    print(f"[+] Evaluation table saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
