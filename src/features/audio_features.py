import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class FeatureExtractor:
    """
    Speech Feature Extraction Engine.
    Computes both model-ready features (e.g. Log Mel-Spectrogram)
    and academic analytical features (Waveform, Spectrogram, MFCC, RMS Energy, ZCR).
    All hyper-parameters are parameter-driven via config.
    """

    def __init__(self, config: Dict[str, Any]):
        feat_cfg = config.get("features", {})
        self.feature_type = feat_cfg.get("feature_type", "melspectrogram")
        self.sample_rate = feat_cfg.get("sample_rate", 16000)
        self.n_fft = feat_cfg.get("n_fft", 1024)
        self.win_length = feat_cfg.get("win_length", 400)
        self.hop_length = feat_cfg.get("hop_length", 160)
        self.n_mels = feat_cfg.get("n_mels", 64)
        self.n_mfcc = feat_cfg.get("n_mfcc", 40)
        self.f_min = feat_cfg.get("f_min", 20.0)
        self.f_max = feat_cfg.get("f_max", 8000.0)
        self.log_mel = feat_cfg.get("log_mel", True)

        # Precompute Mel Filterbank
        self.mel_filterbank = self._build_mel_filterbank()

    def _hz_to_mel(self, hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> np.ndarray:
        """Constructs triangular Mel filterbank matrix (n_mels, n_fft // 2 + 1)."""
        num_bins = self.n_fft // 2 + 1
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        bin_points = np.clip(bin_points, 0, num_bins - 1)

        filterbank = np.zeros((self.n_mels, num_bins), dtype=np.float32)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            if f_m > f_m_minus:
                filterbank[m - 1, f_m_minus:f_m] = (
                    np.arange(f_m_minus, f_m) - f_m_minus
                ) / (f_m - f_m_minus)
            if f_m_plus > f_m:
                filterbank[m - 1, f_m:f_m_plus] = (
                    f_m_plus - np.arange(f_m, f_m_plus)
                ) / (f_m_plus - f_m)

        return filterbank

    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Computes Short-Time Fourier Transform using Hann window.
        Input: audio shape (samples,) or (1, samples)
        Output: Linear Spectrogram complex matrix (n_fft // 2 + 1, num_frames)
        """
        signal = audio.squeeze()
        num_samples = len(signal)
        window = np.hanning(self.win_length)

        num_frames = 1 + (num_samples - self.win_length) // self.hop_length
        if num_frames <= 0:
            num_frames = 1
            padded = np.zeros(self.win_length, dtype=np.float32)
            padded[:min(num_samples, self.win_length)] = signal[:min(num_samples, self.win_length)]
            stft_matrix = np.fft.rfft(padded * window, n=self.n_fft)[:, np.newaxis]
            return stft_matrix

        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(num_frames, self.win_length),
            strides=(signal.strides[0] * self.hop_length, signal.strides[0]),
        )
        windowed_frames = frames * window
        stft_matrix = np.fft.rfft(windowed_frames, n=self.n_fft, axis=-1).T
        return stft_matrix

    def compute_linear_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Returns power spectrogram |STFT|^2."""
        stft = self.compute_stft(audio)
        power_spec = np.abs(stft) ** 2
        return power_spec

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Computes Mel-spectrogram: Filterbank * Power Spectrogram.
        If log_mel is True, returns log(Mel + 1e-6).
        Output shape: (n_mels, num_frames)
        """
        power_spec = self.compute_linear_spectrogram(audio)
        mel_spec = np.dot(self.mel_filterbank, power_spec)
        if self.log_mel:
            mel_spec = np.log(np.maximum(mel_spec, 1e-6))
        return mel_spec.astype(np.float32)

    def compute_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Computes Discrete Cosine Transform (DCT-II) over log Mel-spectrogram
        to extract MFCC features.
        """
        log_mel = self.compute_mel_spectrogram(audio)
        # Compute DCT-II across Mel bands
        n_m = log_mel.shape[0]
        n_mfcc = min(self.n_mfcc, n_m)
        k = np.arange(n_mfcc)[:, np.newaxis]
        n = np.arange(n_m)[np.newaxis, :]
        dct_basis = np.cos(np.pi * k * (2 * n + 1) / (2.0 * n_m))
        mfcc = np.dot(dct_basis, log_mel)
        return mfcc.astype(np.float32)

    @staticmethod
    def compute_rms_energy(audio: np.ndarray, frame_length: int = 400, hop_length: int = 160) -> np.ndarray:
        """Computes frame-wise Root Mean Square (RMS) energy."""
        signal = audio.squeeze()
        num_samples = len(signal)
        num_frames = 1 + max(0, (num_samples - frame_length) // hop_length)
        if num_frames == 0:
            return np.array([np.sqrt(np.mean(signal**2))], dtype=np.float32)
        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(num_frames, frame_length),
            strides=(signal.strides[0] * hop_length, signal.strides[0]),
        )
        rms = np.sqrt(np.mean(frames**2, axis=-1))
        return rms.astype(np.float32)

    @staticmethod
    def compute_zero_crossing_rate(audio: np.ndarray, frame_length: int = 400, hop_length: int = 160) -> np.ndarray:
        """Computes frame-wise Zero Crossing Rate (ZCR)."""
        signal = audio.squeeze()
        num_samples = len(signal)
        num_frames = 1 + max(0, (num_samples - frame_length) // hop_length)
        if num_frames == 0:
            return np.array([0.0], dtype=np.float32)
        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(num_frames, frame_length),
            strides=(signal.strides[0] * hop_length, signal.strides[0]),
        )
        signs = np.sign(frames)
        signs[signs == 0] = 1
        zcr = np.mean(np.abs(np.diff(signs, axis=-1)) > 0, axis=-1)
        return zcr.astype(np.float32)

    def extract_model_input(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extracts the definitive feature tensor to be fed directly into TD-ResNet.
        Returns Tensor of shape (1, n_mels, time_frames).
        """
        if self.feature_type == "mfcc":
            feat = self.compute_mfcc(audio)
        else:
            feat = self.compute_mel_spectrogram(audio)

        # Standard normalization per feature map (z-score)
        mean = np.mean(feat)
        std = np.std(feat) + 1e-6
        feat = (feat - mean) / std

        tensor = torch.from_numpy(feat).unsqueeze(0).float()  # (1, F, T)
        return tensor
