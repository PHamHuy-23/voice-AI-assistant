import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_yaml
from src.data.preprocessing import AudioPreprocessor
from src.features.audio_features import FeatureExtractor
from src.features.visualization import plot_raw_vs_normalized


def main():
    parser = argparse.ArgumentParser(description="Compare RAW vs NORMALIZED audio processing.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file.")
    parser.add_argument("--raw_config", type=str, default="configs/preprocessing/raw.yaml")
    parser.add_argument("--norm_config", type=str, default="configs/preprocessing/normalized.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model/td_resnet.yaml")
    parser.add_argument("--output", type=str, default="reports/figures/raw_vs_normalized.png")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return

    raw_cfg = load_yaml(args.raw_config)
    norm_cfg = load_yaml(args.norm_config)
    feat_cfg = load_yaml(args.model_config)

    raw_prep = AudioPreprocessor(raw_cfg)
    norm_prep = AudioPreprocessor(norm_cfg)
    feat_ext = FeatureExtractor(feat_cfg)

    # Process RAW
    raw_audio, raw_sr, raw_meta = raw_prep.process(args.audio)
    raw_mel = feat_ext.compute_mel_spectrogram(raw_audio)
    raw_energy = feat_ext.compute_rms_energy(raw_audio)

    # Process NORMALIZED
    norm_audio, norm_sr, norm_meta = norm_prep.process(args.audio)
    norm_mel = feat_ext.compute_mel_spectrogram(norm_audio)
    norm_energy = feat_ext.compute_rms_energy(norm_audio)

    print(f"[*] RAW Audio: shape={raw_audio.shape}, sr={raw_sr}, peak={np.max(np.abs(raw_audio)):.4f}")
    print(f"[*] NORMALIZED Audio: shape={norm_audio.shape}, sr={norm_sr}, peak={np.max(np.abs(norm_audio)):.4f}")

    plot_raw_vs_normalized(
        raw_audio=raw_audio,
        raw_sr=raw_sr,
        norm_audio=norm_audio,
        norm_sr=norm_sr,
        raw_mel=raw_mel,
        norm_mel=norm_mel,
        raw_energy=raw_energy,
        norm_energy=norm_energy,
        title=f"Report 2.5: RAW vs NORMALIZED Comparison ({os.path.basename(args.audio)})",
        save_path=args.output,
    )
    print(f"[+] Comparative figure successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
