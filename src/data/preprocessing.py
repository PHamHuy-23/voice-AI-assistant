import wave
import numpy as np
from typing import Dict, Any, Tuple, Optional


class AudioPreprocessor:
    """
    Modular Audio Preprocessing Pipeline.
    Supports granular toggling of operations to facilitate RAW vs. NORMALIZED experiments.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get("mode", "normalized")
        pipeline_cfg = config.get("pipeline", {})

        # Toggles
        self.do_to_mono = pipeline_cfg.get("to_mono", True)
        self.do_resample = pipeline_cfg.get("resample", True)
        self.do_trim_silence = pipeline_cfg.get("trim_silence", True)
        self.do_normalize_amplitude = pipeline_cfg.get("normalize_amplitude", True)
        self.do_pad_or_crop = pipeline_cfg.get("pad_or_crop", True)

        # Numerical thresholds
        self.target_sample_rate = config.get("target_sample_rate", 16000)
        self.target_length_samples = config.get("target_length_samples", 16000)
        self.silence_threshold_db = config.get("silence_threshold_db", 20.0)
        self.norm_type = config.get("norm_type", "peak")
        self.target_peak = config.get("target_peak", 0.95)
        self.pad_mode = config.get("pad_mode", "constant")
        self.pad_position = config.get("pad_position", "center")

    @staticmethod
    def load_wav(file_path: str) -> Tuple[np.ndarray, int]:
        """
        Loads a standard WAV file into float32 numpy array normalized to [-1.0, 1.0].
        Uses standard library wave module for maximum cross-platform reliability.
        """
        with wave.open(file_path, "rb") as wf:
            num_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()
            raw_bytes = wf.readframes(num_frames)

        if sample_width == 1:
            # 8-bit unsigned
            dtype = np.uint8
            audio = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            # 16-bit signed
            dtype = np.int16
            audio = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)
            audio = audio / 32768.0
        elif sample_width == 4:
            # 32-bit signed
            dtype = np.int32
            audio = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)
            audio = audio / 2147483648.0
        else:
            raise ValueError(f"Unsupported bit depth sample width: {sample_width}")

        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).T
        else:
            audio = audio.reshape(1, -1)

        return audio, sample_rate

    @staticmethod
    def to_mono(audio: np.ndarray) -> np.ndarray:
        """Converts multi-channel audio (channels, samples) to mono (1, samples)."""
        if audio.shape[0] > 1:
            mono = np.mean(audio, axis=0, keepdims=True)
            return mono
        return audio

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
        """
        Resamples audio signal using linear interpolation (fast fallback)
        or scipy/librosa when available.
        """
        if orig_sr == target_sr:
            return audio, target_sr

        num_channels, num_samples = audio.shape
        new_num_samples = int(round(num_samples * (target_sr / orig_sr)))

        orig_indices = np.linspace(0, num_samples - 1, num_samples)
        new_indices = np.linspace(0, num_samples - 1, new_num_samples)

        resampled = np.zeros((num_channels, new_num_samples), dtype=np.float32)
        for c in range(num_channels):
            resampled[c] = np.interp(new_indices, orig_indices, audio[c])

        return resampled, target_sr

    @staticmethod
    def trim_silence(audio: np.ndarray, threshold_db: float = 20.0) -> np.ndarray:
        """Trims leading and trailing silence based on energy threshold relative to peak dB."""
        signal = audio[0] if audio.ndim > 1 else audio
        abs_sig = np.abs(signal)
        peak = np.max(abs_sig)
        if peak == 0:
            return audio

        db = 20 * np.log10(np.maximum(abs_sig, 1e-7) / peak)
        active_indices = np.where(db > -threshold_db)[0]

        if len(active_indices) == 0:
            return audio

        start_idx = active_indices[0]
        end_idx = active_indices[-1] + 1
        return audio[:, start_idx:end_idx] if audio.ndim > 1 else audio[start_idx:end_idx]

    @staticmethod
    def normalize_amplitude(audio: np.ndarray, norm_type: str = "peak", target_peak: float = 0.95) -> np.ndarray:
        """Normalizes amplitude using peak normalization or RMS normalization."""
        if norm_type == "peak":
            peak = np.max(np.abs(audio))
            if peak > 1e-7:
                return audio * (target_peak / peak)
            return audio
        elif norm_type == "rms":
            rms = np.sqrt(np.mean(audio**2))
            if rms > 1e-7:
                target_rms = 0.1
                return audio * (target_rms / rms)
            return audio
        return audio

    def pad_or_crop(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pads with zeros or crops audio to exact target length."""
        num_channels, length = audio.shape
        if length == target_length:
            return audio
        elif length > target_length:
            # Crop
            if self.pad_position == "center":
                start = (length - target_length) // 2
                return audio[:, start : start + target_length]
            else:
                return audio[:, :target_length]
        else:
            # Pad
            pad_needed = target_length - length
            if self.pad_position == "center":
                pad_left = pad_needed // 2
                pad_right = pad_needed - pad_left
            else:
                pad_left = 0
                pad_right = pad_needed
            return np.pad(audio, ((0, 0), (pad_left, pad_right)), mode=self.pad_mode)

    def process(self, file_path_or_audio: Any, orig_sr: Optional[int] = None) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Executes the configured preprocessing steps.
        Returns (processed_audio, sample_rate, transformation_metadata).
        """
        meta = {"mode": self.mode, "steps_applied": []}

        if isinstance(file_path_or_audio, str):
            audio, sr = self.load_wav(file_path_or_audio)
            meta["original_sr"] = sr
            meta["original_shape"] = list(audio.shape)
        else:
            audio = file_path_or_audio
            sr = orig_sr or self.target_sample_rate
            meta["original_sr"] = sr
            meta["original_shape"] = list(audio.shape)

        # 1. Convert to Mono
        if self.do_to_mono:
            audio = self.to_mono(audio)
            meta["steps_applied"].append("to_mono")

        # 2. Resample
        if self.do_resample and sr != self.target_sample_rate:
            audio, sr = self.resample(audio, sr, self.target_sample_rate)
            meta["steps_applied"].append(f"resample_{meta['original_sr']}_to_{self.target_sample_rate}")

        # 3. Trim Silence
        if self.do_trim_silence:
            audio = self.trim_silence(audio, threshold_db=self.silence_threshold_db)
            meta["steps_applied"].append("trim_silence")

        # 4. Normalize Amplitude
        if self.do_normalize_amplitude:
            audio = self.normalize_amplitude(audio, norm_type=self.norm_type, target_peak=self.target_peak)
            meta["steps_applied"].append(f"normalize_{self.norm_type}")

        # 5. Pad or Crop
        if self.do_pad_or_crop:
            audio = self.pad_or_crop(audio, target_length=self.target_length_samples)
            meta["steps_applied"].append(f"pad_or_crop_to_{self.target_length_samples}")

        meta["final_sr"] = sr
        meta["final_shape"] = list(audio.shape)
        return audio, sr, meta
