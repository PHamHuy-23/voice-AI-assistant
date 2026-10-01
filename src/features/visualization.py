import os
from typing import Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI compatibility
import matplotlib.pyplot as plt


def plot_feature_gallery(
    waveform: np.ndarray,
    spec: np.ndarray,
    mel: np.ndarray,
    mfcc: np.ndarray,
    energy: np.ndarray,
    zcr: np.ndarray,
    sr: int,
    title: str = "Speech Feature Analysis",
    save_path: Optional[str] = None,
) -> None:
    """
    Renders a comprehensive 6-panel feature grid for academic speech processing reporting (Report 2.4).
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    sig = waveform.squeeze()
    time_axis = np.linspace(0, len(sig) / sr, len(sig))

    # 1. Waveform
    axes[0, 0].plot(time_axis, sig, color="#1f77b4", linewidth=0.8)
    axes[0, 0].set_title("Time-Domain Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Linear Spectrogram
    im1 = axes[0, 1].imshow(np.log1p(spec), aspect="auto", origin="lower", cmap="magma")
    axes[0, 1].set_title("Log Power Spectrogram (STFT)")
    axes[0, 1].set_xlabel("Time Frames")
    axes[0, 1].set_ylabel("Frequency Bins")
    fig.colorbar(im1, ax=axes[0, 1], format="%+2.0f dB")

    # 3. Mel-Spectrogram
    im2 = axes[1, 0].imshow(mel, aspect="auto", origin="lower", cmap="viridis")
    axes[1, 0].set_title("Log Mel-Spectrogram")
    axes[1, 0].set_xlabel("Time Frames")
    axes[1, 0].set_ylabel("Mel Frequency Bands")
    fig.colorbar(im2, ax=axes[1, 0])

    # 4. MFCC
    im3 = axes[1, 1].imshow(mfcc, aspect="auto", origin="lower", cmap="coolwarm")
    axes[1, 1].set_title("MFCC (Mel-Frequency Cepstral Coefficients)")
    axes[1, 1].set_xlabel("Time Frames")
    axes[1, 1].set_ylabel("MFCC Coefficients")
    fig.colorbar(im3, ax=axes[1, 1])

    # 5. RMS Energy
    axes[2, 0].plot(energy, color="#2ca02c", linewidth=1.5)
    axes[2, 0].set_title("Short-Time RMS Energy")
    axes[2, 0].set_xlabel("Frame Index")
    axes[2, 0].set_ylabel("RMS Amplitude")
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)

    # 6. Zero Crossing Rate
    axes[2, 1].plot(zcr, color="#d62728", linewidth=1.5)
    axes[2, 1].set_title("Zero Crossing Rate (ZCR)")
    axes[2, 1].set_xlabel("Frame Index")
    axes[2, 1].set_ylabel("Rate")
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_raw_vs_normalized(
    raw_audio: np.ndarray,
    raw_sr: int,
    norm_audio: np.ndarray,
    norm_sr: int,
    raw_mel: np.ndarray,
    norm_mel: np.ndarray,
    raw_energy: np.ndarray,
    norm_energy: np.ndarray,
    title: str = "RAW vs. NORMALIZED Audio Comparison",
    save_path: Optional[str] = None,
) -> None:
    """
    Renders direct comparative visualization for Report 2.5 (Raw vs Normalized Experiment).
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    raw_sig = raw_audio.squeeze()
    norm_sig = norm_audio.squeeze()
    raw_time = np.linspace(0, len(raw_sig) / raw_sr, len(raw_sig))
    norm_time = np.linspace(0, len(norm_sig) / norm_sr, len(norm_sig))

    # Row 1: Waveforms
    axes[0, 0].plot(raw_time, raw_sig, color="#e377c2", linewidth=0.8)
    axes[0, 0].set_title(f"RAW Waveform (sr={raw_sr}Hz, dur={len(raw_sig)/raw_sr:.2f}s)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    axes[0, 1].plot(norm_time, norm_sig, color="#17becf", linewidth=0.8)
    axes[0, 1].set_title(f"NORMALIZED Waveform (sr={norm_sr}Hz, dur={len(norm_sig)/norm_sr:.2f}s)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Normalized Amplitude")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Row 2: Mel-Spectrograms
    im1 = axes[1, 0].imshow(raw_mel, aspect="auto", origin="lower", cmap="viridis")
    axes[1, 0].set_title("RAW Log Mel-Spectrogram")
    axes[1, 0].set_xlabel("Time Frames")
    axes[1, 0].set_ylabel("Mel Bands")
    fig.colorbar(im1, ax=axes[1, 0])

    im2 = axes[1, 1].imshow(norm_mel, aspect="auto", origin="lower", cmap="viridis")
    axes[1, 1].set_title("NORMALIZED Log Mel-Spectrogram")
    axes[1, 1].set_xlabel("Time Frames")
    axes[1, 1].set_ylabel("Mel Bands")
    fig.colorbar(im2, ax=axes[1, 1])

    # Row 3: RMS Energy Curves
    axes[2, 0].plot(raw_energy, color="#ff7f0e", linewidth=1.5)
    axes[2, 0].set_title(f"RAW RMS Energy (Peak: {np.max(raw_energy):.4f})")
    axes[2, 0].set_xlabel("Frames")
    axes[2, 0].set_ylabel("Energy")
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)

    axes[2, 1].plot(norm_energy, color="#2ca02c", linewidth=1.5)
    axes[2, 1].set_title(f"NORMALIZED RMS Energy (Peak: {np.max(norm_energy):.4f})")
    axes[2, 1].set_xlabel("Frames")
    axes[2, 1].set_ylabel("Energy")
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_training_history(history: Dict[str, list], save_path: str) -> None:
    """Plots training and validation episode accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    ax1.set_title("Episodic Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], "r--s", label="Val Accuracy")
    ax2.set_title("Mean Episode Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
