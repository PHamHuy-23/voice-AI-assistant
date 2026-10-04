"""
========================================================================================
 VoiceShortcuts AI — 5-Shot Voice Command Assistant with Dynamic Feedback & History
 Architecture: PyTorch TC-ResNet8 Dilated (Exp 025 - Best Accuracy: 95.40%)
 Metric: 5-Shot L2-Normalized Prototypical Learning + Cosine Angle Metric
 UX Innovations:
   - Instant Processing Flash: Thẻ kết quả nhấp nháy ngay khi bắt được lượt nói mới
   - Auto-Reset Timer (2.5s): Tự động trả về trạng thái chờ nghe, không bị lưu chữ cũ
   - Live Recognition History Feed: Nhật ký hiển thị từng lượt nhận diện kèm mốc thời gian
   - Dynamic Ambient Noise Calibration & Noise Prototype: Chống nhiễu phòng triệt để
========================================================================================
"""

import os
import sys
import time
import math
import json
import shutil
import threading
import wave
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import customtkinter as ctk

try:
    import sounddevice as sd
except ImportError:
    sd = None

# Ensure project root in sys.path
DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DEMO_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.tc_resnet8 import TCResNet8, load_trained_tcresnet8
from demo.codex_runner import CodexRunResult, CodexRunner
from demo.antigravity_runner import AntigravityChatSession

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tcresnet8_clean_weights.pt")
KEYWORDS_DIR = os.path.join(DEMO_DIR, "user_keywords")
METADATA_FILE = os.path.join(KEYWORDS_DIR, "metadata.json")
CODEX_LOG_DIR = os.path.join(DEMO_DIR, "codex_runs")
os.makedirs(KEYWORDS_DIR, exist_ok=True)

# Theme configuration
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ---------------------------------------------------------------------------
# Audio Processing Utilities
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    """Converts Vietnamese or accented string into clean alphanumeric slug."""
    text = text.replace('đ', 'd').replace('Đ', 'D')
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = nfkd.encode('ASCII', 'ignore').decode('utf-8')
    slug = "".join(c if c.isalnum() else "_" for c in ascii_text.lower()).strip("_")
    return slug if slug else f"cmd_{int(time.time())}"


def calculate_rms(audio_data: np.ndarray) -> float:
    """Calculates root-mean-square amplitude."""
    if len(audio_data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))


def center_voice_energy(audio: np.ndarray, target_length: int = 16000) -> np.ndarray:
    """Calculates moving energy envelope and centers speech mass within 1.0s window."""
    audio = audio.flatten().astype(np.float32)

    if len(audio) < target_length:
        total_pad = target_length - len(audio)
        left = total_pad // 2
        return np.pad(audio, (left, total_pad - left), mode="constant")

    frame_size = 320  # 20ms
    num_frames = len(audio) // frame_size
    energies = [
        np.sum(audio[i * frame_size:(i + 1) * frame_size] ** 2)
        for i in range(num_frames)
    ]

    if len(energies) > 0 and max(energies) > 0:
        weights = np.array(energies)
        weights = np.maximum(0, weights - np.median(weights))
        if weights.sum() > 0:
            center_frame = int(np.average(np.arange(len(weights)), weights=weights))
            center_sample = center_frame * frame_size
        else:
            center_sample = int(np.argmax(np.abs(audio)))
    else:
        center_sample = len(audio) // 2

    start = max(0, center_sample - int(0.40 * target_length))
    if start + target_length > len(audio):
        start = len(audio) - target_length

    return audio[start:start + target_length]


def normalize_and_save_wav(audio_data: np.ndarray, wav_path: str, sr: int = 16000):
    """Aligns speech peak energy to 1.0s window, normalizes amplitude, and saves 16-bit WAV."""
    audio = center_voice_energy(audio_data, target_length=sr)

    peak = np.max(np.abs(audio))
    if peak > 0.05:
        audio = (audio / peak) * 0.90

    int16_data = np.clip(audio * 32767.0, -32767.0, 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_data.tobytes())


# ---------------------------------------------------------------------------
# Few-Shot Prototypical Inference Engine with Noise Prototype
# ---------------------------------------------------------------------------
class FewShotEngine:
    def __init__(self):
        self.sr = 16000
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr,
            n_mfcc=40,
            melkwargs={"n_fft": 640, "hop_length": 320, "win_length": 640}
        )
        self.model = load_trained_tcresnet8(CHECKPOINT_PATH)
        self.model.eval()

        self.prototypes: Dict[str, torch.Tensor] = {}
        self.ambient_noise_proto: Optional[torch.Tensor] = None

        self.ambient_noise_rms = 0.050
        self.speech_trigger_rms = 0.095
        self.min_similarity_threshold = 0.70
        self.min_winner_margin = 0.04
        self.prototype_thresholds: Dict[str, float] = {}

        self.update_prototypes()

    def wav_to_normalized_embedding(self, wav_path: str) -> torch.Tensor:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        samples = center_voice_energy(samples, target_length=self.sr)
        wav_t = torch.from_numpy(samples).unsqueeze(0)
        feat = self.mfcc_transform(wav_t)[0].T.unsqueeze(0)  # [1, 51, 40]
        
        with torch.no_grad():
            emb = self.model(feat)
            norm_emb = F.normalize(emb, p=2, dim=-1)
        return norm_emb

    def raw_audio_to_normalized_embedding(self, audio: np.ndarray) -> torch.Tensor:
        samples = center_voice_energy(audio, target_length=self.sr)
        peak = np.max(np.abs(samples))
        if peak > 0.08:
            samples = (samples / peak) * 0.90

        wav_t = torch.from_numpy(samples).unsqueeze(0)
        feat = self.mfcc_transform(wav_t)[0].T.unsqueeze(0)

        with torch.no_grad():
            emb = self.model(feat)
            norm_emb = F.normalize(emb, p=2, dim=-1)
        return norm_emb

    def calibrate_ambient_noise(self, noise_audio: np.ndarray):
        rms = calculate_rms(noise_audio)
        self.ambient_noise_rms = max(0.010, rms)
        self.speech_trigger_rms = max(0.065, self.ambient_noise_rms * 1.85)

        with torch.no_grad():
            self.ambient_noise_proto = self.raw_audio_to_normalized_embedding(noise_audio)

    def update_prototypes(self):
        self.prototypes.clear()
        self.prototype_thresholds.clear()
        if not os.path.exists(KEYWORDS_DIR):
            return

        with torch.no_grad():
            for kw in sorted(os.listdir(KEYWORDS_DIR)):
                kw_dir = os.path.join(KEYWORDS_DIR, kw)
                if not os.path.isdir(kw_dir):
                    continue
                wav_files = sorted([os.path.join(kw_dir, f) for f in os.listdir(kw_dir) if f.endswith(".wav")])
                if not wav_files:
                    continue

                embs = [self.wav_to_normalized_embedding(w) for w in wav_files]
                cat_embs = torch.cat(embs, dim=0)
                proto = cat_embs.mean(dim=0, keepdim=True)
                proto = F.normalize(proto, p=2, dim=-1)
                self.prototypes[kw] = proto
                enrollment_sims = torch.matmul(cat_embs, proto.T).flatten()
                # Use a robust low percentile so one bad extra recording cannot
                # make the command accept almost everything.
                low_percentile = torch.quantile(enrollment_sims, 0.10)
                calibrated = float(low_percentile.item()) - 0.025
                self.prototype_thresholds[kw] = min(0.95, max(0.88, self.min_similarity_threshold, calibrated))

    def classify_audio(self, audio_data: np.ndarray) -> Dict[str, any]:
        if not self.prototypes:
            return {"status": "EMPTY", "message": "Chưa có khẩu lệnh nào được đăng ký!"}

        rms = calculate_rms(audio_data)

        if rms < self.speech_trigger_rms:
            return {"status": "SILENCE", "rms": rms}

        peak = np.max(np.abs(audio_data))
        crest_factor = peak / (rms + 1e-8)
        if crest_factor < 2.5:
            return {"status": "NOISE_REJECTED", "rms": rms, "crest": crest_factor}

        t0 = time.perf_counter()
        with torch.no_grad():
            q_emb = self.raw_audio_to_normalized_embedding(audio_data)

            if self.ambient_noise_proto is not None:
                noise_sim = torch.sum(q_emb * self.ambient_noise_proto).item()
            else:
                noise_sim = -1.0

            sims = {}
            for kw, proto in self.prototypes.items():
                cos_sim = torch.sum(q_emb * proto).item()
                cos_sim = max(-1.0, min(1.0, cos_sim))
                sims[kw] = cos_sim

            best_kw = max(sims, key=sims.get)
            max_sim = sims[best_kw]
            ordered_sims = sorted(sims.values(), reverse=True)
            runner_up_sim = ordered_sims[1] if len(ordered_sims) > 1 else None
            winner_margin = max_sim - runner_up_sim if runner_up_sim is not None else None
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if noise_sim > max_sim and noise_sim > 0.65:
                return {
                    "status": "NOISE_PROTOTYPE_MATCH",
                    "rms": rms,
                    "noise_sim": noise_sim,
                    "latency_ms": latency_ms
                }

            temp = 0.15
            exp_vals = {k: math.exp(s / temp) for k, s in sims.items()}
            sum_exp = sum(exp_vals.values())
            conf_pct = (exp_vals[best_kw] / sum_exp) * 100.0 if sum_exp > 0 else 0.0

            required_similarity = self.prototype_thresholds.get(best_kw, self.min_similarity_threshold)
            similarity_ok = max_sim >= required_similarity
            required_margin = 0.02 if max_sim >= 0.98 else self.min_winner_margin
            margin_ok = winner_margin is None or winner_margin >= required_margin
            is_match = similarity_ok and margin_ok

            return {
                "status": "OK",
                "is_match": is_match,
                "best_keyword": best_kw,
                "similarity": max_sim,
                "required_similarity": required_similarity,
                "winner_margin": winner_margin,
                "required_margin": required_margin,
                "confidence": conf_pct,
                "latency_ms": latency_ms,
                "rms": rms,
                "noise_sim": noise_sim,
                "similarities": sims
            }


# ---------------------------------------------------------------------------
# Create Command Dialog (5-Shot Apple-style Wizard)
# ---------------------------------------------------------------------------
class AddCommandDialog(ctk.CTkToplevel):
    def __init__(self, parent, on_saved_callback):
        super().__init__(parent)
        self.parent = parent
        self.on_saved_callback = on_saved_callback

        self.title("Tạo Khẩu Lệnh 5-Shot — VoiceShortcuts")
        self.geometry("600x740")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self.recorded_samples: List[np.ndarray] = []
        self.is_recording = False
        self.sr = 16000
        self.sample_target_count = 5

        self._build_ui()

    def _build_ui(self):
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=24, pady=(20, 10))

        title = ctk.CTkLabel(
            header_frame,
            text="✨ Tạo Voice Command (5-Shot)",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            header_frame,
            text="Thu âm chính xác 5 lần để mô hình đạt độ ổn định tối đa (Bão hòa 5-Shot).",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8"
        )
        subtitle.pack(anchor="w", pady=(4, 0))

        form_frame = ctk.CTkFrame(self, fg_color="#1e293b", corner_radius=12)
        form_frame.pack(fill="x", padx=24, pady=10)

        lbl_name = ctk.CTkLabel(form_frame, text="TÊN KHẨU LỆNH (COMMAND NAME):", font=ctk.CTkFont(size=11, weight="bold"), text_color="#cbd5e1")
        lbl_name.pack(anchor="w", padx=16, pady=(14, 4))

        self.entry_name = ctk.CTkEntry(
            form_frame,
            placeholder_text="Ví dụ: Bật đèn, Mở trình duyệt, Khóa máy...",
            height=38,
            font=ctk.CTkFont(size=13)
        )
        self.entry_name.pack(fill="x", padx=16, pady=(0, 10))

        lbl_desc = ctk.CTkLabel(form_frame, text="YÊU CẦU GỬI CHO CODEX CLI (PROMPT / ACTION):", font=ctk.CTkFont(size=11, weight="bold"), text_color="#cbd5e1")
        lbl_desc.pack(anchor="w", padx=16, pady=(4, 4))

        self.entry_desc = ctk.CTkEntry(
            form_frame,
            placeholder_text="Ví dụ: Kiểm tra dự án, sửa lỗi test và ghi lại kết quả",
            height=38,
            font=ctk.CTkFont(size=13)
        )
        self.entry_desc.pack(fill="x", padx=16, pady=(0, 16))

        record_frame = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=12, border_width=1, border_color="#334155")
        record_frame.pack(fill="x", padx=24, pady=10)

        step_title = ctk.CTkLabel(
            record_frame,
            text="🎙️ THU ÂM HỌC MẪU (5 LẦN PHÁT ÂM LIÊN TIẾP)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8"
        )
        step_title.pack(anchor="w", padx=16, pady=(14, 10))

        self.badges_frame = ctk.CTkFrame(record_frame, fg_color="transparent")
        self.badges_frame.pack(fill="x", padx=14, pady=4)

        self.badge_labels = []
        for i in range(self.sample_target_count):
            badge = ctk.CTkLabel(
                self.badges_frame,
                text=f"Mẫu #{i+1}\nChưa thu",
                fg_color="#334155",
                corner_radius=8,
                height=42,
                font=ctk.CTkFont(size=10, weight="bold")
            )
            badge.pack(side="left", padx=3, expand=True, fill="x")
            self.badge_labels.append(badge)

        self.lbl_record_status = ctk.CTkLabel(
            record_frame,
            text="Sẵn sàng thu mẫu 1. Nhấn nút bên dưới và phát âm rõ từ khóa.",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8"
        )
        self.lbl_record_status.pack(padx=16, pady=(12, 6))

        self.record_progress = ctk.CTkProgressBar(record_frame, height=6)
        self.record_progress.set(0)
        self.record_progress.pack(fill="x", padx=20, pady=(0, 14))

        self.btn_record_sample = ctk.CTkButton(
            record_frame,
            text="🔴 Bắt Đầu Thu Mẫu #1 (1.3 Giây)",
            command=self._start_recording_sample,
            height=44,
            fg_color="#dc2626",
            hover_color="#b91c1c",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.btn_record_sample.pack(fill="x", padx=16, pady=(0, 16))

        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill="x", padx=24, pady=(14, 20), side="bottom")

        self.btn_cancel = ctk.CTkButton(
            bottom_frame,
            text="Hủy Bỏ",
            command=self.destroy,
            width=110,
            height=40,
            fg_color="#475569",
            hover_color="#334155"
        )
        self.btn_cancel.pack(side="left")

        self.btn_save = ctk.CTkButton(
            bottom_frame,
            text="💾 Lưu Khẩu Lệnh 5-Shot",
            command=self._save_command,
            state="disabled",
            height=40,
            fg_color="#10b981",
            hover_color="#059669",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.btn_save.pack(side="right", fill="x", expand=True, padx=(12, 0))

    def _start_recording_sample(self):
        if self.is_recording:
            return
        if sd is None:
            self.lbl_record_status.configure(text="❌ Lỗi: sounddevice chưa sẵn sàng.", text_color="#ef4444")
            return

        cmd_name = self.entry_name.get().strip()
        if not cmd_name:
            self.lbl_record_status.configure(text="⚠️ Vui lòng nhập Tên khẩu lệnh trước khi thu âm!", text_color="#f59e0b")
            return

        self.is_recording = True
        self.btn_record_sample.configure(state="disabled")

        idx = len(self.recorded_samples) + 1
        self.lbl_record_status.configure(text=f"🎙️ Đang thu mẫu #{idx}/{self.sample_target_count}... Hãy nói từ khóa ngay!", text_color="#38bdf8")

        def record_thread():
            duration = 1.3
            sr = 16000
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')

            steps = 20
            for s in range(steps):
                time.sleep(duration / steps)
                self.record_progress.set((s + 1) / steps)

            sd.wait()
            audio_flat = audio.flatten()
            self.recorded_samples.append(audio_flat)
            self.is_recording = False

            self.after(0, self._on_sample_recorded)

        threading.Thread(target=record_thread, daemon=True).start()

    def _on_sample_recorded(self):
        count = len(self.recorded_samples)
        self.record_progress.set(0)

        self.badge_labels[count - 1].configure(
            text=f"Mẫu #{count}\nĐã thu ✓",
            fg_color="#059669",
            text_color="#ffffff"
        )

        if count < self.sample_target_count:
            next_idx = count + 1
            self.lbl_record_status.configure(
                text=f"✓ Đã lưu mẫu #{count}. Sẵn sàng thu mẫu #{next_idx}/{self.sample_target_count}.",
                text_color="#10b981"
            )
            self.btn_record_sample.configure(
                text=f"🔴 Thu Tiếp Mẫu #{next_idx} (1.3 Giây)",
                state="normal"
            )
        else:
            self.lbl_record_status.configure(
                text="🎉 Đã thu đủ 5/5 mẫu! Nhấn 'Lưu Khẩu Lệnh' để hoàn tất.",
                text_color="#10b981"
            )
            self.btn_record_sample.configure(
                text="✓ Đã Đủ 5 Mẫu",
                state="disabled",
                fg_color="#334155"
            )
            self.btn_save.configure(state="normal")

    def _save_command(self):
        cmd_name = self.entry_name.get().strip()
        cmd_desc = self.entry_desc.get().strip()

        if not cmd_desc:
            self.lbl_record_status.configure(
                text="⚠️ Vui lòng nhập yêu cầu cụ thể để Codex thực hiện.",
                text_color="#f59e0b"
            )
            return
        if not cmd_name or len(self.recorded_samples) < self.sample_target_count:
            return

        slug = slugify(cmd_name)
        target_dir = os.path.join(KEYWORDS_DIR, slug)
        os.makedirs(target_dir, exist_ok=True)

        for i, audio in enumerate(self.recorded_samples[:self.sample_target_count]):
            wav_path = os.path.join(target_dir, f"sample_{i+1}.wav")
            normalize_and_save_wav(audio, wav_path, sr=self.sr)

        meta = {}
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        meta[slug] = {
            "name": cmd_name,
            "description": cmd_desc,
            "action_type": "codex",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "samples_count": self.sample_target_count
        }

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if self.on_saved_callback:
            self.on_saved_callback()

        self.destroy()


# ---------------------------------------------------------------------------
# Main VoiceShortcuts Application Window (Feedback Flash & History Feed)
# ---------------------------------------------------------------------------
class VoiceShortcutsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Voice — AI Assistant")
        self.geometry("1200x820")
        self.minsize(1040, 720)

        # Engine
        self.engine = FewShotEngine()

        # Audio Stream states
        self.is_listening = True
        self.stream_handle = None
        self.sr = 16000
        self.stream_buffer = np.zeros(int(1.3 * self.sr), dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.last_detection_time = 0.0
        self.detection_cooldown_s = 1.4
        self.is_calibrating = False
        self.is_testing_manual = False
        self.is_in_background = False
        self.codex_runner = CodexRunner(PROJECT_ROOT, CODEX_LOG_DIR)
        self.codex_last_started = {}
        self.codex_command_cooldown_s = 10.0
        self.chat_session = None
        self.chat_busy = False
        self.chat_response_label = None
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Reset Timer for Card
        self.reset_timer_id = None
        self.detection_counter = 0

        # Build UI
        self._build_ui()

        # Load keywords
        self.refresh_keywords_list()

        # Start background continuous audio stream
        self.start_audio_stream()

        # Auto calibrate room noise floor after 0.5s
        self.after(500, self._auto_calibrate_room_noise)

    def hide_to_background(self):
        """Hides window to run silently in background while keeping mic listening 24/7."""
        self.is_in_background = True
        self.withdraw()
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONINFORMATION)
        except Exception:
            pass
        print("\n" + "=" * 65)
        print(" 🪟 [VoiceShortcuts] ĐÃ CHUYỂN SANG CHẾ ĐỘ CHẠY ẨN NỀN.")
        print(" 🎙️ Microphone vẫn đang tiếp tục lắng nghe liên tục.")
        print(" 🍏 Khi bạn phát âm 'Hey Siri', cửa sổ sẽ tự động bật lên màn hình!")
        print("=" * 65 + "\n")

    def wakeup_to_foreground(self):
        """Restores window from background or minimized state to frontmost desktop focus."""
        self.is_in_background = False
        try:
            self.deiconify()
            self.state("normal")
            self.lift()
            self.attributes("-topmost", True)
            self.after(400, lambda: self.attributes("-topmost", False))
            self.focus_force()

            if sys.platform == "win32":
                import ctypes
                hwnd = ctypes.windll.user32.FindWindowW(None, self.title())
                if not hwnd:
                    try:
                        hwnd = self.winfo_id()
                        parent = ctypes.windll.user32.GetParent(hwnd)
                        if parent:
                            hwnd = parent
                    except Exception:
                        pass
                if hwnd:
                    ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE = 9
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception as e:
            print(f"[VoiceShortcuts] Lỗi wakeup: {e}")

    def _play_siri_chime(self):
        """Plays signature Apple Siri two-tone chime in background thread."""
        def chime():
            try:
                import winsound
                # Tone 1: E5 (659Hz, 110ms) -> Tone 2: A5 (880Hz, 170ms)
                winsound.Beep(659, 110)
                winsound.Beep(880, 170)
            except Exception:
                pass
        threading.Thread(target=chime, daemon=True).start()

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # =====================================================================
        # LEFT SIDEBAR: Commands Management
        # =====================================================================
        sidebar = ctk.CTkFrame(self, width=330, corner_radius=0, fg_color=("#f2f2f7", "#111111"))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(2, weight=1)

        # App Logo & Title
        title_box = ctk.CTkFrame(sidebar, fg_color="transparent")
        title_box.grid(row=0, column=0, padx=20, pady=(24, 14), sticky="w")

        lbl_logo = ctk.CTkLabel(title_box, text="Voice", font=ctk.CTkFont(size=24, weight="bold"), text_color=("#111111", "#f5f5f7"))
        lbl_logo.pack(anchor="w")

        lbl_sub = ctk.CTkLabel(title_box, text="Personal AI Assistant", font=ctk.CTkFont(size=12), text_color=("#6e6e73", "#a1a1a6"))
        lbl_sub.pack(anchor="w", pady=(2, 0))

        # Button + Tạo Command Mới (5-Shot)
        self.btn_add_cmd = ctk.CTkButton(
            sidebar,
            text="＋  Command mới",
            command=self._open_add_command_dialog,
            height=44,
            corner_radius=10,
            fg_color=("#111111", "#f5f5f7"),
            hover_color=("#333333", "#d2d2d7"),
            text_color=("#ffffff", "#111111"),
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.btn_add_cmd.grid(row=1, column=0, padx=20, pady=(4, 14), sticky="ew")

        # Scrollable list of commands
        lbl_list_title = ctk.CTkLabel(
            sidebar,
            text="DANH SÁCH KHẨU LỆNH ĐÃ LƯU:",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#94a3b8"
        )
        lbl_list_title.grid(row=2, column=0, padx=24, pady=(0, 4), sticky="nw")

        self.scroll_cmds = ctk.CTkScrollableFrame(sidebar, fg_color="transparent")
        self.scroll_cmds.grid(row=2, column=0, padx=12, pady=(24, 10), sticky="nsew")

        # Noise & Sensitivity Calibration Box
        settings_box = ctk.CTkFrame(sidebar, fg_color="#1e293b", corner_radius=10)
        settings_box.grid(row=3, column=0, padx=16, pady=16, sticky="ew")

        self.btn_calibrate = ctk.CTkButton(
            settings_box,
            text="🎯 Hiệu Chuẩn Tiếng Ồn Phòng (1s)",
            command=self._manual_calibrate_noise,
            height=32,
            fg_color="#334155",
            hover_color="#475569",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.btn_calibrate.pack(fill="x", padx=12, pady=(10, 6))

        self.lbl_noise_info = ctk.CTkLabel(
            settings_box,
            text=f"Ồn phòng: {self.engine.ambient_noise_rms:.3f} | Bắt giọng: >{self.engine.speech_trigger_rms:.3f}",
            font=ctk.CTkFont(size=10),
            text_color="#94a3b8"
        )
        self.lbl_noise_info.pack(padx=12, pady=(0, 6))

        lbl_thresh = ctk.CTkLabel(
            settings_box,
            text="Ngưỡng Nhận Diện Độ Tương Đồng (Cosine):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#cbd5e1"
        )
        lbl_thresh.pack(anchor="w", padx=12, pady=(6, 2))

        self.lbl_thresh_val = ctk.CTkLabel(
            settings_box,
            text=f"Tương đồng >= {self.engine.min_similarity_threshold * 100:.0f}% (Chống nhận nhầm)",
            font=ctk.CTkFont(size=10),
            text_color="#38bdf8"
        )
        self.lbl_thresh_val.pack(anchor="w", padx=12, pady=(0, 4))

        self.slider_thresh = ctk.CTkSlider(
            settings_box,
            from_=0.50,
            to=0.90,
            number_of_steps=40,
            command=self._on_threshold_changed,
            height=14
        )
        self.slider_thresh.set(self.engine.min_similarity_threshold)
        self.slider_thresh.pack(fill="x", padx=12, pady=(0, 8))

        # Wake Word & Background status banner
        bg_info_box = ctk.CTkFrame(settings_box, fg_color="#1e1b4b", corner_radius=8, border_width=1, border_color="#4f46e5")
        bg_info_box.pack(fill="x", padx=10, pady=(4, 10))

        lbl_bg_title = ctk.CTkLabel(
            bg_info_box,
            text="🍏 Wake Word: 'Hey Siri' (Đánh Thức)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#a5b4fc"
        )
        lbl_bg_title.pack(anchor="w", padx=8, pady=(4, 1))

        lbl_bg_desc = ctk.CTkLabel(
            bg_info_box,
            text="Khi bấm 'Chạy Ẩn Nền', app sẽ thu nhỏ và tiếp tục lắng nghe 24/7. Nói 'Hey Siri' để tự động mở cửa sổ lên!",
            font=ctk.CTkFont(size=9),
            text_color="#c7d2fe",
            wraplength=290,
            justify="left"
        )
        lbl_bg_desc.pack(anchor="w", padx=8, pady=(0, 6))

        # =====================================================================
        # RIGHT PANEL: Live Listening Monitor & Real-time Display
        # =====================================================================
        main_panel = ctk.CTkFrame(self, fg_color=("#ffffff", "#000000"))
        main_panel.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        main_panel.grid_rowconfigure(2, weight=1)
        main_panel.grid_rowconfigure(3, weight=1)
        main_panel.grid_columnconfigure(0, weight=1)

        # Header of Main Panel
        header = ctk.CTkFrame(main_panel, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 14))

        panel_title = ctk.CTkLabel(
            header,
            text="Voice Assistant",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        panel_title.pack(side="left")

        right_header_btns = ctk.CTkFrame(header, fg_color="transparent")
        right_header_btns.pack(side="right")

        self.view_switch = ctk.CTkSegmentedButton(
            right_header_btns,
            values=["Voice", "Chat"],
            command=self._show_view,
            fg_color=("#e5e5ea", "#1c1c1e"),
            selected_color=("#111111", "#f5f5f7"),
            selected_hover_color=("#333333", "#d2d2d7"),
            unselected_color=("#e5e5ea", "#1c1c1e"),
            unselected_hover_color=("#d2d2d7", "#2c2c2e"),
            text_color=("#111111", "#111111"),
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.view_switch.set("Voice")
        self.view_switch.pack(side="left", padx=(0, 12))

        self.btn_theme = ctk.CTkButton(
            right_header_btns,
            text="◐",
            width=34,
            height=32,
            command=self._toggle_theme,
            fg_color=("#e5e5ea", "#1c1c1e"),
            hover_color=("#d2d2d7", "#2c2c2e"),
            text_color=("#111111", "#f5f5f7"),
        )
        self.btn_theme.pack(side="left", padx=(0, 10))

        self.btn_bg_mode = ctk.CTkButton(
            right_header_btns,
            text="🪟 Chạy Ẩn Nền",
            command=self.hide_to_background,
            height=32,
            fg_color="#4f46e5",
            hover_color="#4338ca",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.btn_bg_mode.pack(side="left", padx=(0, 10))

        self.btn_manual_test = ctk.CTkButton(
            right_header_btns,
            text="🎙️ Bấm Để Nói (Test 1.2s)",
            command=self._run_manual_test,
            height=32,
            fg_color="#475569",
            hover_color="#334155",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.btn_manual_test.pack(side="left", padx=(0, 14))

        self.switch_listen = ctk.CTkSwitch(
            right_header_btns,
            text="Lắng Nghe Liên Tục",
            command=self._toggle_listening,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.switch_listen.select()
        self.switch_listen.pack(side="left")

        # Live Waveform / VU Meter Box
        vu_box = ctk.CTkFrame(main_panel, fg_color="#0f172a", corner_radius=12, border_width=1, border_color="#1e293b")
        vu_box.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 14))

        self.lbl_mic_status = ctk.CTkLabel(
            vu_box,
            text="🟢 Đang lắng nghe microphone... Phòng yên tĩnh.",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#10b981"
        )
        self.lbl_mic_status.pack(anchor="w", padx=16, pady=(10, 4))

        self.vu_bar = ctk.CTkProgressBar(vu_box, height=10, fg_color="#1e293b", progress_color="#38bdf8")
        self.vu_bar.set(0)
        self.vu_bar.pack(fill="x", padx=16, pady=(4, 12))

        # Real-time Result Display Card (Prominent Card)
        self.result_card = ctk.CTkFrame(
            main_panel,
            fg_color="#0f172a",
            corner_radius=16,
            border_width=2,
            border_color="#334155"
        )
        self.result_card.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 14))
        self.result_card.grid_columnconfigure(0, weight=1)

        self.lbl_res_icon = ctk.CTkLabel(
            self.result_card,
            text="👂",
            font=ctk.CTkFont(size=48)
        )
        self.lbl_res_icon.pack(pady=(22, 6))

        self.lbl_res_title = ctk.CTkLabel(
            self.result_card,
            text="Đang chờ giọng nói tiếp theo...",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#94a3b8"
        )
        self.lbl_res_title.pack(pady=2)

        self.lbl_res_desc = ctk.CTkLabel(
            self.result_card,
            text="Hãy phát âm to và rõ một khẩu lệnh. Thẻ sẽ tự động làm mới sau mỗi lượt nói.",
            font=ctk.CTkFont(size=12),
            text_color="#64748b"
        )
        self.lbl_res_desc.pack(pady=2)

        # Metric stats badge row
        self.stats_frame = ctk.CTkFrame(self.result_card, fg_color="transparent")
        self.stats_frame.pack(pady=(12, 16))

        self.stat_sim = ctk.CTkLabel(
            self.stats_frame,
            text="Độ Tương Đồng: --",
            fg_color="#1e293b",
            corner_radius=8,
            width=160,
            height=32,
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.stat_sim.pack(side="left", padx=5)

        self.stat_conf = ctk.CTkLabel(
            self.stats_frame,
            text="Độ Tin Cậy: --",
            fg_color="#1e293b",
            corner_radius=8,
            width=160,
            height=32,
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.stat_conf.pack(side="left", padx=5)

        self.stat_latency = ctk.CTkLabel(
            self.stats_frame,
            text="Độ Trễ: --",
            fg_color="#1e293b",
            corner_radius=8,
            width=160,
            height=32,
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.stat_latency.pack(side="left", padx=5)

        # =====================================================================
        # LOWER AREA: Live Recognition History Feed
        # =====================================================================
        history_box = ctk.CTkFrame(main_panel, fg_color="#0f172a", corner_radius=12, border_width=1, border_color="#1e293b")
        history_box.grid(row=3, column=0, sticky="nsew", padx=10, pady=0)
        history_box.grid_rowconfigure(1, weight=1)
        history_box.grid_columnconfigure(0, weight=1)

        hist_header = ctk.CTkFrame(history_box, fg_color="transparent")
        hist_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 4))

        lbl_hist_title = ctk.CTkLabel(
            hist_header,
            text="📋 NHẬT KÝ NHẬN DIỆN THỜI GIAN THỰC (DETECTION HISTORY):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#94a3b8"
        )
        lbl_hist_title.pack(side="left")

        btn_clear_hist = ctk.CTkButton(
            hist_header,
            text="Xóa nhật ký",
            command=self._clear_history,
            width=80,
            height=22,
            fg_color="#334155",
            hover_color="#475569",
            font=ctk.CTkFont(size=10)
        )
        btn_clear_hist.pack(side="right")

        self.scroll_history = ctk.CTkScrollableFrame(history_box, fg_color="transparent")
        self.scroll_history.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))

        self.lbl_hist_empty = ctk.CTkLabel(
            self.scroll_history,
            text="Chưa có lượt nhận diện nào. Hãy nói một từ khóa để bắt đầu ghi nhật ký.",
            font=ctk.CTkFont(size=11),
            text_color="#475569"
        )
        self.lbl_hist_empty.pack(pady=18)

        self.voice_frames = [vu_box, self.result_card, history_box]
        self._build_chat_panel(main_panel)

    def _build_chat_panel(self, parent):
        self.chat_panel = ctk.CTkFrame(parent, fg_color="transparent")
        self.chat_panel.grid(row=1, column=0, rowspan=3, sticky="nsew", padx=10, pady=(0, 2))
        self.chat_panel.grid_rowconfigure(1, weight=1)
        self.chat_panel.grid_columnconfigure(0, weight=1)

        chat_header = ctk.CTkFrame(
            self.chat_panel,
            fg_color=("#f2f2f7", "#111111"),
            corner_radius=16,
        )
        chat_header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        chat_header.grid_columnconfigure(0, weight=1)

        chat_title = ctk.CTkLabel(
            chat_header,
            text="Antigravity",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=("#111111", "#f5f5f7"),
        )
        chat_title.grid(row=0, column=0, padx=18, pady=(12, 1), sticky="w")
        self.lbl_chat_status = ctk.CTkLabel(
            chat_header,
            text="Chưa kết nối · phiên persistent stream-json",
            font=ctk.CTkFont(size=11),
            text_color=("#6e6e73", "#a1a1a6"),
        )
        self.lbl_chat_status.grid(row=1, column=0, padx=18, pady=(0, 12), sticky="w")

        self.switch_agent_tools = ctk.CTkSwitch(
            chat_header,
            text="Cho phép hành động",
            font=ctk.CTkFont(size=11),
            width=130,
            progress_color=("#111111", "#f5f5f7"),
            button_color=("#ffffff", "#111111"),
        )
        self.switch_agent_tools.grid(row=0, column=1, rowspan=2, padx=18, pady=12)

        self.chat_messages = ctk.CTkScrollableFrame(
            self.chat_panel,
            fg_color=("#ffffff", "#000000"),
            corner_radius=0,
        )
        self.chat_messages.grid(row=1, column=0, sticky="nsew")
        self.chat_messages.grid_columnconfigure(0, weight=1)
        self._add_chat_bubble(
            "assistant",
            "Chào bạn. Mình là Antigravity và sẽ giữ nguyên phiên trò chuyện để phản hồi nhanh hơn sau tin nhắn đầu tiên.",
        )

        composer = ctk.CTkFrame(
            self.chat_panel,
            fg_color=("#f2f2f7", "#1c1c1e"),
            corner_radius=18,
        )
        composer.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        composer.grid_columnconfigure(0, weight=1)

        self.chat_input = ctk.CTkTextbox(
            composer,
            height=68,
            corner_radius=14,
            border_width=0,
            fg_color="transparent",
            font=ctk.CTkFont(size=13),
            wrap="word",
        )
        self.chat_input.grid(row=0, column=0, padx=(14, 6), pady=8, sticky="ew")
        self.chat_input.bind("<Control-Return>", lambda _event: self._send_chat_message())

        self.btn_chat_send = ctk.CTkButton(
            composer,
            text="↑",
            width=42,
            height=42,
            corner_radius=21,
            command=self._send_chat_message,
            fg_color=("#111111", "#f5f5f7"),
            hover_color=("#333333", "#d2d2d7"),
            text_color=("#ffffff", "#111111"),
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.btn_chat_send.grid(row=0, column=1, padx=(4, 12), pady=12)
        self.chat_panel.grid_remove()

    def _show_view(self, value):
        if value == "Chat":
            for frame in self.voice_frames:
                frame.grid_remove()
            self.chat_panel.grid()
            self.chat_input.focus_set()
        else:
            self.chat_panel.grid_remove()
            for frame in self.voice_frames:
                frame.grid()

    def _toggle_theme(self):
        current = ctk.get_appearance_mode().lower()
        ctk.set_appearance_mode("light" if current == "dark" else "dark")

    def _add_chat_bubble(self, role: str, text: str):
        is_user = role == "user"
        row = ctk.CTkFrame(self.chat_messages, fg_color="transparent")
        row.grid(sticky="ew", padx=8, pady=5)
        row.grid_columnconfigure(0, weight=1)
        bubble = ctk.CTkLabel(
            row,
            text=text,
            justify="left",
            anchor="w",
            wraplength=620,
            font=ctk.CTkFont(size=13),
            fg_color=(("#111111" if is_user else "#e5e5ea"), ("#f5f5f7" if is_user else "#1c1c1e")),
            text_color=(("#ffffff" if is_user else "#111111"), ("#111111" if is_user else "#f5f5f7")),
            corner_radius=16,
            padx=14,
            pady=10,
        )
        bubble.grid(row=0, column=0, sticky="e" if is_user else "w", padx=(100, 0) if is_user else (0, 100))
        self.after(20, lambda: self.chat_messages._parent_canvas.yview_moveto(1.0))
        return bubble

    def _send_chat_message(self):
        message = self.chat_input.get("1.0", "end").strip()
        if not message or self.chat_busy:
            return "break"
        self.chat_input.delete("1.0", "end")
        self._add_chat_bubble("user", message)
        self.chat_response_label = self._add_chat_bubble("assistant", "Đang suy nghĩ…")
        self.chat_busy = True
        self.btn_chat_send.configure(state="disabled")

        if self.chat_session is None:
            self.chat_session = AntigravityChatSession(
                PROJECT_ROOT,
                allow_unattended_tools=bool(self.switch_agent_tools.get()),
                on_event=lambda event: self.after(0, lambda e=event: self._on_chat_event(e)),
                on_error=lambda error: self.after(0, lambda e=error: self._on_chat_error(e)),
            )
            self.switch_agent_tools.configure(state="disabled")
        try:
            self.chat_session.send(message)
            self.lbl_chat_status.configure(text="Đang kết nối Antigravity…")
        except Exception as exc:
            self._on_chat_error(str(exc))
        return "break"

    @staticmethod
    def _chat_event_text(event):
        for key in ("text_delta", "delta", "text", "content", "message"):
            value = event.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for nested in ("text", "content", "message"):
                    if isinstance(value.get(nested), str):
                        return value[nested]
        return ""

    def _on_chat_event(self, event):
        event_type = str(event.get("event") or event.get("type") or "").lower()
        if event_type in {"init", "ready", "session_started"}:
            self.lbl_chat_status.configure(text="Đã kết nối · phiên đang hoạt động")
            return
        text = self._chat_event_text(event)
        if text and self.chat_response_label is not None:
            current = self.chat_response_label.cget("text")
            current = "" if current == "Đang suy nghĩ…" else current
            self.chat_response_label.configure(text=current + text)
        if event_type in {"result", "done", "turn_completed", "completed"}:
            self.chat_busy = False
            self.btn_chat_send.configure(state="normal")
            self.lbl_chat_status.configure(text="Đã kết nối · sẵn sàng")

    def _on_chat_error(self, error):
        self.chat_busy = False
        self.btn_chat_send.configure(state="normal")
        self.lbl_chat_status.configure(text="Mất kết nối")
        if self.chat_response_label is not None:
            self.chat_response_label.configure(text=f"Không thể kết nối Antigravity: {error}")

    def _on_close(self):
        if self.chat_session is not None:
            self.chat_session.close()
        if self.stream_handle is not None:
            try:
                self.stream_handle.stop()
                self.stream_handle.close()
            except Exception:
                pass
        self.destroy()

    def _auto_calibrate_room_noise(self):
        threading.Thread(target=self._calibration_worker, daemon=True).start()

    def _manual_calibrate_noise(self):
        if self.is_calibrating:
            return
        threading.Thread(target=self._calibration_worker, daemon=True).start()

    def _calibration_worker(self):
        self.is_calibrating = True
        self.after(0, lambda: self.btn_calibrate.configure(text="⏳ Đang đo ồn phòng...", state="disabled"))
        self.after(0, lambda: self.lbl_mic_status.configure(text="🎯 Giữ im lặng 1 giây để hệ thống đo tiếng ồn phòng...", text_color="#38bdf8"))

        duration = 1.0
        rec = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1, dtype='float32')
        sd.wait()
        noise_audio = rec.flatten()

        self.engine.calibrate_ambient_noise(noise_audio)
        self.is_calibrating = False

        self.after(0, lambda: self.btn_calibrate.configure(text="🎯 Đo Lại Tiếng Ồn Phòng (1s)", state="normal"))
        self.after(0, lambda: self.lbl_noise_info.configure(
            text=f"Ồn phòng: {self.engine.ambient_noise_rms:.3f} | Bắt giọng: >{self.engine.speech_trigger_rms:.3f}"
        ))
        self.after(0, lambda: self.lbl_mic_status.configure(text="🟢 Đã cân chỉnh tiếng ồn phòng! Đang lắng nghe...", text_color="#10b981"))

    def _on_threshold_changed(self, value):
        val = float(value)
        self.engine.min_similarity_threshold = val
        self.lbl_thresh_val.configure(text=f"Tương đồng >= {val * 100:.0f}% (Chống nhận nhầm)")

    def _toggle_listening(self):
        self.is_listening = self.switch_listen.get()
        if self.is_listening:
            self.lbl_mic_status.configure(text="🟢 Đang lắng nghe microphone...", text_color="#10b981")
        else:
            self.lbl_mic_status.configure(text="⏸️ Đã tạm dừng lắng nghe.", text_color="#f59e0b")
            self.vu_bar.set(0)

    def _run_manual_test(self):
        if self.is_testing_manual:
            return
        if not self.engine.prototypes:
            self.lbl_mic_status.configure(text="⚠️ Vui lòng tạo ít nhất 1 khẩu lệnh trước khi test!", text_color="#f59e0b")
            return

        self.is_testing_manual = True
        self.btn_manual_test.configure(state="disabled", text="🎙️ Đang nghe...")
        self.lbl_mic_status.configure(text="🔴 Đang ghi âm thử nghiệm (1.2s)... Nói ngay!", text_color="#ef4444")

        # Instant visual feedback
        self._flash_processing_state()

        def test_worker():
            duration = 1.2
            audio = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1, dtype='float32')
            sd.wait()
            self.is_testing_manual = False
            res = self.engine.classify_audio(audio.flatten())
            self.after(0, lambda: self.btn_manual_test.configure(state="normal", text="🎙️ Bấm Để Nói (Test 1.2s)"))
            self.after(0, lambda: self.lbl_mic_status.configure(text="🟢 Đang lắng nghe microphone...", text_color="#10b981"))
            if res.get("status") == "OK":
                self.after(0, lambda: self._update_detection_result(res))
            elif res.get("status") in ["SILENCE", "NOISE_REJECTED", "NOISE_PROTOTYPE_MATCH"]:
                self.after(0, lambda: self._update_noise_result(res))

        threading.Thread(target=test_worker, daemon=True).start()

    def _open_add_command_dialog(self):
        AddCommandDialog(self, on_saved_callback=self.refresh_keywords_list)

    def refresh_keywords_list(self):
        for child in self.scroll_cmds.winfo_children():
            child.destroy()

        self.engine.update_prototypes()

        meta = {}
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        if not meta:
            empty_lbl = ctk.CTkLabel(
                self.scroll_cmds,
                text="Chưa có khẩu lệnh nào.\nNhấn '+ Tạo Command Mới (5-Shot)'\nđể thu 5 mẫu âm thanh.",
                font=ctk.CTkFont(size=12),
                text_color="#64748b"
            )
            empty_lbl.pack(pady=40)
            return

        for slug, data in meta.items():
            card = ctk.CTkFrame(self.scroll_cmds, fg_color="#1e293b", corner_radius=10)
            card.pack(fill="x", pady=5, padx=2)

            top_row = ctk.CTkFrame(card, fg_color="transparent")
            top_row.pack(fill="x", padx=12, pady=(10, 2))

            name_lbl = ctk.CTkLabel(
                top_row,
                text=f"🗣️ {data.get('name', slug)}",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#f8fafc"
            )
            name_lbl.pack(side="left")

            btn_del = ctk.CTkButton(
                top_row,
                text="✕",
                width=26,
                height=26,
                fg_color="#334155",
                hover_color="#ef4444",
                font=ctk.CTkFont(size=11, weight="bold"),
                command=lambda s=slug: self._delete_command(s)
            )
            btn_del.pack(side="right")

            desc_lbl = ctk.CTkLabel(
                card,
                text=f"{data.get('description', '')} • {'Codex CLI' if data.get('action_type') == 'codex' else 'Chưa gán hành động'} • 5-Shot ✓",
                font=ctk.CTkFont(size=11),
                text_color="#94a3b8",
                wraplength=260,
                justify="left"
            )
            desc_lbl.pack(anchor="w", padx=12, pady=(0, 8))

    def _delete_command(self, slug: str):
        target_dir = os.path.join(KEYWORDS_DIR, slug)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if slug in meta:
                    del meta[slug]
                with open(METADATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        self.refresh_keywords_list()

    def start_audio_stream(self):
        if sd is None:
            self.lbl_mic_status.configure(text="❌ sounddevice không khả dụng.", text_color="#ef4444")
            return

        def audio_callback(indata, frames, time_info, status):
            if not self.is_listening:
                return
            audio_chunk = indata[:, 0]
            with self.buffer_lock:
                self.stream_buffer = np.roll(self.stream_buffer, -len(audio_chunk))
                self.stream_buffer[-len(audio_chunk):] = audio_chunk

        try:
            self.stream_handle = sd.InputStream(
                samplerate=self.sr,
                channels=1,
                blocksize=1024,
                dtype="float32",
                callback=audio_callback
            )
            self.stream_handle.start()
        except Exception as e:
            self.lbl_mic_status.configure(text=f"❌ Không thể mở Microphone: {e}", text_color="#ef4444")
            return

        threading.Thread(target=self._stream_analysis_worker, daemon=True).start()

    def _stream_analysis_worker(self):
        while True:
            time.sleep(0.08)
            if not self.is_listening or not self.engine.prototypes or self.is_testing_manual or self.is_calibrating:
                continue

            with self.buffer_lock:
                current_audio = self.stream_buffer.copy()

            rms = calculate_rms(current_audio)
            norm_vu = min(1.0, rms / (self.engine.speech_trigger_rms * 2.5 + 1e-6))
            self.after(0, lambda r=norm_vu: self.vu_bar.set(r))

            now = time.time()
            if rms >= self.engine.speech_trigger_rms and (now - self.last_detection_time > self.detection_cooldown_s):
                # Flash UI immediately: We heard something!
                self.after(0, self._flash_processing_state)

                res = self.engine.classify_audio(current_audio)
                if res.get("status") == "OK":
                    self.last_detection_time = now
                    self.after(0, lambda r=res: self._update_detection_result(r))
                elif res.get("status") in ["NOISE_REJECTED", "NOISE_PROTOTYPE_MATCH"]:
                    self.last_detection_time = now
                    self.after(0, lambda r=res: self._update_noise_result(r))

    def _flash_processing_state(self):
        """Instant visual pulse showing a new speech chunk is being classified."""
        self.result_card.configure(border_color="#38bdf8", fg_color="#0c4a6e")
        self.lbl_res_icon.configure(text="⚡")
        self.lbl_res_title.configure(text="Đang phân tích lượt nói mới...", text_color="#7dd3fc")

    def _schedule_card_reset(self, delay_ms=2500):
        """Auto-resets the main card back to neutral after delay, so it never stays stuck."""
        if self.reset_timer_id is not None:
            self.after_cancel(self.reset_timer_id)
        self.reset_timer_id = self.after(delay_ms, self._reset_card_to_neutral)

    def _reset_card_to_neutral(self):
        self.result_card.configure(border_color="#334155", fg_color="#0f172a")
        self.lbl_res_icon.configure(text="👂")
        self.lbl_res_title.configure(
            text="Đang chờ giọng nói tiếp theo...",
            text_color="#94a3b8"
        )
        self.lbl_res_desc.configure(
            text="Hãy phát âm to và rõ một khẩu lệnh đã lưu.",
            text_color="#64748b"
        )
        self.stat_sim.configure(text="Độ Tương Đồng: --", text_color="#f8fafc")
        self.stat_conf.configure(text="Độ Tin Cậy: --", text_color="#f8fafc")
        self.stat_latency.configure(text="Độ Trễ: --", text_color="#f8fafc")

    def _add_history_entry(self, icon: str, title: str, subtitle: str, color: str):
        """Adds a new row to the detection history feed."""
        if self.lbl_hist_empty.winfo_ismapped():
            self.lbl_hist_empty.pack_forget()

        self.detection_counter += 1
        now_str = datetime.now().strftime("%H:%M:%S")

        row = ctk.CTkFrame(self.scroll_history, fg_color="#1e293b", corner_radius=8, height=32)
        row.pack(fill="x", pady=2, padx=4)

        lbl_badge = ctk.CTkLabel(
            row,
            text=f"#{self.detection_counter} • {now_str}",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#64748b"
        )
        lbl_badge.pack(side="left", padx=(10, 8))

        lbl_text = ctk.CTkLabel(
            row,
            text=f"{icon} {title}  |  {subtitle}",
            font=ctk.CTkFont(size=11),
            text_color=color
        )
        lbl_text.pack(side="left", padx=4)

    def _clear_history(self):
        for child in self.scroll_history.winfo_children():
            child.destroy()
        self.detection_counter = 0
        self.lbl_hist_empty = ctk.CTkLabel(
            self.scroll_history,
            text="Đã xóa lịch sử. Đang chờ các lượt nhận diện tiếp theo...",
            font=ctk.CTkFont(size=11),
            text_color="#475569"
        )
        self.lbl_hist_empty.pack(pady=18)

    def _update_detection_result(self, res: dict):
        is_match = res.get("is_match", False)
        best_kw = res.get("best_keyword", "")
        sim = res.get("similarity", 0.0)
        conf = res.get("confidence", 0.0)
        latency = res.get("latency_ms", 0.0)

        cmd_name = best_kw
        cmd_desc = ""
        meta = {}
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if best_kw in meta:
                    cmd_name = meta[best_kw].get("name", best_kw)
                    cmd_desc = meta[best_kw].get("description", "")
            except Exception:
                pass

        thresh_pct = self.engine.min_similarity_threshold * 100.0
        is_siri = ("siri" in best_kw.lower()) or ("siri" in cmd_name.lower())

        if is_match:
            if is_siri:
                # 🍏 SIRI WAKE-UP TRIGGERED!
                self.wakeup_to_foreground()
                self._play_siri_chime()

                self.result_card.configure(border_color="#38bdf8", fg_color="#082f49")
                self.lbl_res_icon.configure(text="🍏")
                self.lbl_res_title.configure(
                    text="SIRI: ĐÃ ĐÁNH THỨC CỬA SỔ!",
                    text_color="#38bdf8"
                )
                self.lbl_res_desc.configure(
                    text=f"Khẩu lệnh '{cmd_name}' (Độ tương đồng {sim * 100:.1f}%) ➜ Ứng dụng đã tự động mở lên màn hình chính.",
                    text_color="#bae6fd"
                )
                self.stat_sim.configure(text=f"Tương đồng: {sim * 100:.1f}% ✓", text_color="#38bdf8")
                self.stat_conf.configure(text=f"Độ tin cậy: {conf:.1f}%", text_color="#38bdf8")
                self.stat_latency.configure(text=f"Độ trễ: {latency:.1f} ms", text_color="#38bdf8")

                self._add_history_entry("🍏", f"WAKE-UP SIRI: '{cmd_name}'", f"Đã đánh thức màn hình • {sim * 100:.1f}% • {latency:.1f}ms", "#38bdf8")
            else:
                if self.is_in_background:
                    self.wakeup_to_foreground()

                # ✅ RECOGNIZED CORRECTLY
                self.result_card.configure(border_color="#10b981", fg_color="#064e3b")
                self.lbl_res_icon.configure(text="🎯")
                self.lbl_res_title.configure(
                    text=f"ĐÃ NHẬN DIỆN: {cmd_name}",
                    text_color="#34d399"
                )
                self.lbl_res_desc.configure(
                    text=f"Hành động: {cmd_desc} • Tương đồng {sim * 100:.1f}% >= {thresh_pct:.0f}%",
                    text_color="#a7f3d0"
                )
                self.stat_sim.configure(text=f"Tương đồng: {sim * 100:.1f}% ✓", text_color="#34d399")
                self.stat_conf.configure(text=f"Độ tin cậy: {conf:.1f}%", text_color="#34d399")
                self.stat_latency.configure(text=f"Độ trễ: {latency:.1f} ms", text_color="#34d399")

                # Add to history feed
                self._add_history_entry("🎯", f"KHỚP LỆNH: '{cmd_name}'", f"Tương đồng {sim * 100:.1f}% • {latency:.1f}ms", "#34d399")
                self._trigger_codex_action(best_kw, cmd_name, cmd_desc, meta.get(best_kw, {}))

        else:
            # ⚠️ REJECTED: UNKNOWN WORD / WRONG COMMAND
            self.result_card.configure(border_color="#f59e0b", fg_color="#451a03")
            self.lbl_res_icon.configure(text="⚠️")
            self.lbl_res_title.configure(
                text="TỪ LẠ / KHÔNG KHỚP LỆNH NÀO",
                text_color="#fbbf24"
            )
            self.lbl_res_desc.configure(
                text=f"Độ tương đồng chỉ đạt {sim * 100:.1f}% < {thresh_pct:.0f}% (Dưới ngưỡng cho phép) -> Bị từ chối.",
                text_color="#fde68a"
            )
            self.stat_sim.configure(text=f"Tương đồng: {sim * 100:.1f}% ✗", text_color="#fbbf24")
            self.stat_conf.configure(text=f"Độ tin cậy: {conf:.1f}%", text_color="#fbbf24")
            self.stat_latency.configure(text=f"Độ trễ: {latency:.1f} ms", text_color="#fbbf24")

            # Add to history feed
            self._add_history_entry("⚠️", "TỪ LẠ / NÓI SAI", f"Tương đồng {sim * 100:.1f}% < {thresh_pct:.0f}%", "#fbbf24")

        # Automatically schedule card reset after 2.5s so user knows system is ready for next command!
        self._schedule_card_reset(delay_ms=2500)

    def _trigger_codex_action(self, slug: str, name: str, prompt: str, metadata: dict):
        """Starts the configured Codex prompt once recognition is accepted."""
        if metadata.get("action_type") != "codex" or not prompt.strip():
            return

        now = time.time()
        last_started = self.codex_last_started.get(slug, 0.0)
        if now - last_started < self.codex_command_cooldown_s:
            self._add_history_entry("⏳", f"CODEX: '{name}'", "Bỏ qua kích hoạt lặp trong 10 giây", "#f59e0b")
            return

        def on_started(_slug):
            self.codex_last_started[slug] = time.time()
            self.after(0, lambda: self._add_history_entry("🤖", f"CODEX ĐANG CHẠY: '{name}'", prompt[:90], "#a78bfa"))
            self.after(0, lambda: self.lbl_mic_status.configure(
                text=f"🤖 Codex đang thực hiện: {name}", text_color="#a78bfa"
            ))

        def on_finished(result: CodexRunResult):
            self.after(0, lambda r=result: self._on_codex_finished(name, r))

        started = self.codex_runner.start(slug, prompt, on_started, on_finished)
        if not started:
            active = self.codex_runner.active_slug
            self._add_history_entry("⏳", f"CODEX ĐANG BẬN: '{name}'", f"Đang xử lý lệnh {active}", "#f59e0b")

    def _on_codex_finished(self, name: str, result: CodexRunResult):
        if result.success:
            summary = result.output.replace("\n", " ").strip()[-140:] or "Hoàn thành không có thông báo."
            self._add_history_entry("✅", f"CODEX HOÀN THÀNH: '{name}'", summary, "#34d399")
            self.lbl_mic_status.configure(text="🟢 Codex đã hoàn thành. Tiếp tục lắng nghe...", text_color="#10b981")
        else:
            detail = result.error.replace("\n", " ").strip()[-140:] or f"Mã lỗi {result.exit_code}"
            self._add_history_entry("❌", f"CODEX THẤT BẠI: '{name}'", detail, "#ef4444")
            self.lbl_mic_status.configure(text=f"❌ Codex lỗi. Xem log: {result.log_path}", text_color="#ef4444")

    def _update_noise_result(self, res: dict):
        status = res.get("status", "")
        self.result_card.configure(border_color="#334155", fg_color="#0f172a")
        if status == "SILENCE":
            self.lbl_res_icon.configure(text="🤫")
            self.lbl_res_title.configure(text="Khoảng Lặng / Tiếng Nhỏ", text_color="#94a3b8")
            self.lbl_res_desc.configure(text=f"Âm lượng ({res.get('rms', 0):.3f}) dưới ngưỡng bắt giọng.", text_color="#64748b")
        else:
            self.lbl_res_icon.configure(text="🛡️")
            self.lbl_res_title.configure(text="Đã Lọc Tiếng Ồn Môi Trường", text_color="#38bdf8")
            self.lbl_res_desc.configure(text="Tín hiệu phát hiện là tiếng ồn liên tục -> Đã loại bỏ.", text_color="#94a3b8")
            self._add_history_entry("🛡️", "LỌC TIẾNG ỒN", "Tiếng quạt / xì mic được loại bỏ", "#94a3b8")

        self._schedule_card_reset(delay_ms=1800)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = VoiceShortcutsApp()
    app.mainloop()
