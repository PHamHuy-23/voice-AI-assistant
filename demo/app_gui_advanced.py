"""
========================================================================================
 Voice AI Assistant — Modern Desktop GUI (Few-Shot Keyword Spotting)
 Architecture: Official PyTorch TC-ResNet8 + Prototypical Networks (64,560 params)
 Checkpoint: Exp 025 (Test Acc: 95.40%, Loss: 0.1240, D=48 Embedding)
 UI Framework: CustomTkinter (Modern Dark Glassmorphism Aesthetic)
========================================================================================
"""

import os
import sys
import time
import math
import struct
import wave
import subprocess
import threading
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import customtkinter as ctk

# Ensure project root in sys.path
DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DEMO_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.tc_resnet8 import TCResNet8, load_trained_tcresnet8

SUPPORT_DIR = os.path.join(DEMO_DIR, "support")
SAMPLES_DIR = os.path.join(DEMO_DIR, "samples")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tcresnet8_clean_weights.pt")

# Configure CustomTkinter theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

KEYWORDS = {
    "open_notepad": {
        "title": "Mở Notepad",
        "icon": "📝",
        "freq": 440,
        "action": "Khởi chạy trình soạn thảo Notepad",
        "cmd": "notepad.exe",
        "desc": "Mở ứng dụng soạn thảo văn bản Windows"
    },
    "open_browser": {
        "title": "Mở Web / GitHub",
        "icon": "🌐",
        "freq": 660,
        "action": "Mở trình duyệt Web mặc định",
        "cmd": "explorer.exe https://github.com",
        "desc": "Mở trang web GitHub trên trình duyệt"
    },
    "system_mute": {
        "title": "Tắt / Bật Âm lượng",
        "icon": "🔇",
        "freq": 880,
        "action": "Tắt / Bật nhanh âm lượng loa",
        "cmd": "powershell -c (New-Object -ComObject Wscript.Shell).SendKeys([char]173)",
        "desc": "Gửi tín hiệu Windows Mute để đảo trạng thái âm thanh"
    },
    "stop_task": {
        "title": "Dừng Tác vụ",
        "icon": "⏹️",
        "freq": 330,
        "action": "Dừng tác vụ / Đóng ứng dụng",
        "cmd": None,
        "desc": "Phát thông điệp hủy tiến trình hoặc đóng cửa sổ"
    },
}


def ensure_samples():
    """Ensures fallback synthetic audio files exist."""
    os.makedirs(SUPPORT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    sr = 16000

    def create_wav(fp, base_freq, noise_level=0.04):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with wave.open(fp, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            frames = bytearray()
            for i in range(sr):
                t = i / sr
                sig = 0.6 * math.sin(2 * math.pi * base_freq * t) + 0.25 * math.sin(3 * math.pi * base_freq * t)
                noise = (math.sin(i * 12.9898) % 1.0 - 0.5) * noise_level
                val = int(max(-32767, min(32767, (sig + noise) * 16000)))
                frames.extend(struct.pack("<h", val))
            wf.writeframes(frames)

    for kw, meta in KEYWORDS.items():
        kw_dir = os.path.join(SUPPORT_DIR, kw)
        for s in range(1, 4):
            fp = os.path.join(kw_dir, f"sample_{s}.wav")
            if not os.path.exists(fp):
                create_wav(fp, meta["freq"] * (1.0 + (s - 2) * 0.03))
        q_fp = os.path.join(SAMPLES_DIR, f"query_{kw}.wav")
        if not os.path.exists(q_fp):
            create_wav(q_fp, meta["freq"] * 1.01)

    unk_fp = os.path.join(SAMPLES_DIR, "query_unknown.wav")
    if not os.path.exists(unk_fp):
        create_wav(unk_fp, 1800.0, noise_level=0.45)


class InferenceEngine:
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
        self.confidence_threshold = 75.0
        self.max_distance_threshold = 1.10
        self.enroll_all()

    def extract_mfcc(self, wav_path: str) -> torch.Tensor:
        with wave.open(wav_path, "r") as wf:
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Pad or crop to 16,000 samples
        if len(samples) < self.sr:
            samples = np.pad(samples, (0, self.sr - len(samples)))
        else:
            samples = samples[:self.sr]

        wav_t = torch.from_numpy(samples).unsqueeze(0)
        feat = self.mfcc_transform(wav_t)[0].T
        return feat.unsqueeze(0)  # [1, 51, 40]

    def enroll_all(self):
        """Computes class prototypes from support sets."""
        self.prototypes.clear()
        with torch.no_grad():
            for kw in sorted(os.listdir(SUPPORT_DIR)):
                kw_dir = os.path.join(SUPPORT_DIR, kw)
                if not os.path.isdir(kw_dir):
                    continue
                wavs = [os.path.join(kw_dir, f) for f in os.listdir(kw_dir) if f.endswith(".wav")]
                if not wavs:
                    continue
                embs = [self.model(self.extract_mfcc(w)) for w in wavs]
                cat_embs = torch.cat(embs, dim=0)
                proto = F.normalize(cat_embs.mean(dim=0, keepdim=True), p=2, dim=1)
                self.prototypes[kw] = proto

    def classify(self, wav_path: str) -> Dict[str, any]:
        t0 = time.perf_counter()
        with torch.no_grad():
            feat = self.extract_mfcc(wav_path)
            q_emb = self.model(feat)

            dists = {kw: torch.sum((q_emb - proto) ** 2).item() for kw, proto in self.prototypes.items()}
            temperature = 0.1
            exp_d = {kw: math.exp(-d / temperature) for kw, d in dists.items()}
            sum_exp = sum(exp_d.values())
            probs = {kw: (exp_d[kw] / sum_exp) * 100.0 for kw in dists}

            best_kw = min(dists, key=dists.get)
            min_dist = dists[best_kw]
            conf = probs[best_kw]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        is_rejected = (min_dist > self.max_distance_threshold or conf < self.confidence_threshold)

        return {
            "predicted": "UNKNOWN_REJECTED" if is_rejected else best_kw,
            "raw_best": best_kw,
            "confidence": conf,
            "min_dist": min_dist,
            "distances": dists,
            "latency_ms": latency_ms,
            "is_rejected": is_rejected
        }


class VoiceAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Voice AI Assistant — Few-Shot Keyword Spotting (TC-ResNet8 + ProtoNet)")
        self.geometry("1100 × 740".replace("×", "x"))
        self.minsize(980, 680)

        # Initialize engine & audio samples
        ensure_samples()
        self.engine = InferenceEngine()
        self.execute_real = False

        self._build_ui()

    def _build_ui(self):
        # Grid layout: Header (Row 0), Body (Row 1), Footer (Row 2)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ================= 1. HEADER =================
        header_frame = ctk.CTkFrame(self, fg_color="#1e293b", corner_radius=12, border_width=1, border_color="#334155")
        header_frame.grid(row=0, column=0, padx=16, pady=(14, 10), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="🎙️ DESKTOP VOICE AI ASSISTANT — FEW-SHOT KEYWORD SPOTTING",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#38bdf8"
        )
        title_lbl.grid(row=0, column=0, padx=16, pady=(8, 2), sticky="w")

        sub_lbl = ctk.CTkLabel(
            header_frame,
            text="Backbone: PyTorch TC-ResNet8 (64,560 tham số) • Trọng số tối ưu Exp 025 (Acc: 95.40%) • Embedding D=48",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8"
        )
        sub_lbl.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="w")

        # Status badge
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="● HỆ THỐNG SẴN SÀNG | 4 KHẨU LỆNH ĐÃ NẠP",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#10b981",
            fg_color="#064e3b",
            corner_radius=8,
            padx=10,
            pady=4
        )
        self.status_badge.grid(row=0, column=1, rowspan=2, padx=16, pady=8, sticky="e")

        # ================= 2. MAIN BODY (2 COLUMNS) =================
        body_frame = ctk.CTkFrame(self, fg_color="transparent")
        body_frame.grid(row=1, column=0, padx=16, pady=4, sticky="nsew")
        body_frame.grid_columnconfigure(0, weight=4)  # Left panel: Enrollment (40%)
        body_frame.grid_columnconfigure(1, weight=6)  # Right panel: Inference (60%)
        body_frame.grid_rowconfigure(0, weight=1)

        # ----- LEFT PANEL: ENROLLMENT -----
        left_panel = ctk.CTkFrame(body_frame, fg_color="#1e293b", corner_radius=12, border_width=1, border_color="#334155")
        left_panel.grid(row=0, column=0, padx=(0, 8), pady=0, sticky="nsew")
        left_panel.grid_columnconfigure(0, weight=1)

        left_title = ctk.CTkLabel(
            left_panel,
            text="📋 DANH MỤC KHẨU LỆNH (ENROLLMENT)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#e2e8f0"
        )
        left_title.grid(row=0, column=0, padx=14, pady=(12, 6), sticky="w")

        left_desc = ctk.CTkLabel(
            left_panel,
            text="Mỗi khẩu lệnh chỉ cần 3 mẫu âm thanh (3-shot) để tính Prototype c_k mà không cần huấn luyện lại mạng.",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
            wraplength=340,
            justify="left"
        )
        left_desc.grid(row=1, column=0, padx=14, pady=(0, 10), sticky="w")

        # Keyword list cards
        self.kw_cards_frame = ctk.CTkScrollableFrame(left_panel, fg_color="transparent")
        self.kw_cards_frame.grid(row=2, column=0, padx=10, pady=4, sticky="nsew")
        left_panel.grid_rowconfigure(2, weight=1)

        self._render_keyword_cards()

        # Enrollment buttons
        enroll_btn_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        enroll_btn_frame.grid(row=3, column=0, padx=12, pady=12, sticky="ew")
        enroll_btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_live_enroll = ctk.CTkButton(
            enroll_btn_frame,
            text="🎙️ Thu âm giọng thật (3 lần)",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._on_click_enroll_live
        )
        self.btn_live_enroll.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        self.btn_reset_samples = ctk.CTkButton(
            enroll_btn_frame,
            text="🔄 Nạp lại mẫu chuẩn",
            font=ctk.CTkFont(size=12),
            fg_color="#475569",
            hover_color="#334155",
            command=self._on_click_reset_samples
        )
        self.btn_reset_samples.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        # ----- RIGHT PANEL: TESTING & DISPATCHER -----
        right_panel = ctk.CTkFrame(body_frame, fg_color="#1e293b", corner_radius=12, border_width=1, border_color="#334155")
        right_panel.grid(row=0, column=1, padx=(8, 0), pady=0, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(3, weight=1)

        right_title = ctk.CTkLabel(
            right_panel,
            text="⚡ ĐIỀU KHIỂN & SUY LUẬN THỜI GIAN THỰC",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#e2e8f0"
        )
        right_title.grid(row=0, column=0, padx=16, pady=(12, 4), sticky="w")

        # Mode switch bar
        mode_frame = ctk.CTkFrame(right_panel, fg_color="#0f172a", corner_radius=8)
        mode_frame.grid(row=1, column=0, padx=14, pady=4, sticky="ew")
        mode_frame.grid_columnconfigure(0, weight=1)

        self.switch_exec = ctk.CTkSwitch(
            mode_frame,
            text="Thực thi thật lệnh hệ thống trên Windows (Mở Notepad / Mở Web / Mute loa)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#f1f5f9",
            progress_color="#10b981",
            command=self._on_toggle_exec
        )
        self.switch_exec.grid(row=0, column=0, padx=12, pady=8, sticky="w")

        # Action Buttons Area
        actions_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        actions_frame.grid(row=2, column=0, padx=14, pady=6, sticky="ew")
        actions_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        # Big Live Microphone Button
        self.btn_mic = ctk.CTkButton(
            actions_frame,
            text="🎙️ BẤM VÀ NÓI VÀO MICRO (1S)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#8b5cf6",
            hover_color="#7c3aed",
            height=40,
            command=self._on_click_live_mic
        )
        self.btn_mic.grid(row=0, column=0, columnspan=5, padx=2, pady=(0, 8), sticky="ew")

        # Quick test buttons
        test_buttons = [
            ("📝 Test Notepad", "query_open_notepad.wav", "#3b82f6"),
            ("🌐 Test Web", "query_open_browser.wav", "#06b6d4"),
            ("🔇 Test Mute", "query_system_mute.wav", "#10b981"),
            ("⏹️ Test Stop", "query_stop_task.wav", "#f59e0b"),
            ("⚠️ Test Tạp âm", "query_unknown.wav", "#ef4444"),
        ]
        for col_idx, (b_title, q_file, b_color) in enumerate(test_buttons):
            btn = ctk.CTkButton(
                actions_frame,
                text=b_title,
                font=ctk.CTkFont(size=11),
                fg_color=b_color,
                hover_color="#1e293b",
                height=30,
                command=lambda f=q_file: self._on_click_test_sample(f)
            )
            btn.grid(row=1, column=col_idx, padx=2, pady=2, sticky="ew")

        # Detection Result Card (Centerpiece)
        result_box = ctk.CTkFrame(right_panel, fg_color="#0f172a", corner_radius=10, border_width=1, border_color="#334155")
        result_box.grid(row=3, column=0, padx=14, pady=8, sticky="nsew")
        result_box.grid_columnconfigure(0, weight=1)

        res_header = ctk.CTkLabel(
            result_box,
            text="KẾT QUẢ SUY LUẬN & ĐỐI CHIẾU METRIC PROTOTYPE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#94a3b8"
        )
        res_header.grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")

        # Large Decision Text
        self.lbl_decision = ctk.CTkLabel(
            result_box,
            text="CHƯA CÓ TRUY VẤN",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#cbd5e1"
        )
        self.lbl_decision.grid(row=1, column=0, padx=12, pady=4)

        # Meta metrics (Confidence & Latency)
        meta_subframe = ctk.CTkFrame(result_box, fg_color="transparent")
        meta_subframe.grid(row=2, column=0, padx=12, pady=2, sticky="ew")
        meta_subframe.grid_columnconfigure((0, 1), weight=1)

        self.lbl_conf = ctk.CTkLabel(
            meta_subframe,
            text="Độ tin cậy: --%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8"
        )
        self.lbl_conf.grid(row=0, column=0, sticky="w")

        self.lbl_latency = ctk.CTkLabel(
            meta_subframe,
            text="Độ trễ PyTorch: -- ms",
            font=ctk.CTkFont(size=12),
            text_color="#a855f7"
        )
        self.lbl_latency.grid(row=0, column=1, sticky="e")

        # Confidence Progress Bar
        self.progress_conf = ctk.CTkProgressBar(result_box, progress_color="#38bdf8", fg_color="#334155")
        self.progress_conf.grid(row=3, column=0, padx=12, pady=4, sticky="ew")
        self.progress_conf.set(0.0)

        # Distance Bars Frame
        self.dist_bars_frame = ctk.CTkFrame(result_box, fg_color="transparent")
        self.dist_bars_frame.grid(row=4, column=0, padx=12, pady=6, sticky="ew")
        self.dist_bars_frame.grid_columnconfigure(1, weight=1)

        self.dist_labels = {}
        self.dist_bars = {}
        for row_i, (k, meta) in enumerate(KEYWORDS.items()):
            lbl_name = ctk.CTkLabel(
                self.dist_bars_frame,
                text=f"{meta['icon']} {k}:",
                font=ctk.CTkFont(size=11),
                text_color="#e2e8f0",
                width=130,
                anchor="w"
            )
            lbl_name.grid(row=row_i, column=0, padx=2, pady=2, sticky="w")

            bar = ctk.CTkProgressBar(self.dist_bars_frame, height=10, fg_color="#334155", progress_color="#10b981")
            bar.grid(row=row_i, column=1, padx=6, pady=2, sticky="ew")
            bar.set(0.0)

            lbl_val = ctk.CTkLabel(
                self.dist_bars_frame,
                text="d = --",
                font=ctk.CTkFont(size=10),
                text_color="#94a3b8",
                width=65,
                anchor="e"
            )
            lbl_val.grid(row=row_i, column=2, padx=2, pady=2, sticky="e")

            self.dist_labels[k] = lbl_val
            self.dist_bars[k] = bar

        # Action Execution Notice
        self.lbl_action_status = ctk.CTkLabel(
            result_box,
            text="Trạng thái hành động: Chờ lệnh kích hoạt.",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#94a3b8",
            fg_color="#1e293b",
            corner_radius=6,
            padx=8,
            pady=4
        )
        self.lbl_action_status.grid(row=5, column=0, padx=12, pady=(6, 10), sticky="ew")

        # ================= 3. FOOTER =================
        footer = ctk.CTkLabel(
            self,
            text="Đồ án Xử lý Tiếng nói: Few-Shot Keyword Spotting using TC-ResNet8 & Prototypical Networks • Học kỳ Giữa kỳ 2026",
            font=ctk.CTkFont(size=10),
            text_color="#64748b"
        )
        footer.grid(row=2, column=0, padx=16, pady=(4, 8))

    def _render_keyword_cards(self):
        """Renders list of enrolled keyword cards."""
        for widget in self.kw_cards_frame.winfo_children():
            widget.destroy()

        for kw, meta in KEYWORDS.items():
            card = ctk.CTkFrame(self.kw_cards_frame, fg_color="#0f172a", corner_radius=8, border_width=1, border_color="#334155")
            card.pack(fill="x", padx=4, pady=4)
            card.grid_columnconfigure(0, weight=1)

            t_lbl = ctk.CTkLabel(
                card,
                text=f"{meta['icon']} {kw}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#38bdf8"
            )
            t_lbl.grid(row=0, column=0, padx=8, pady=(4, 0), sticky="w")

            d_lbl = ctk.CTkLabel(
                card,
                text=meta["action"],
                font=ctk.CTkFont(size=10),
                text_color="#94a3b8"
            )
            d_lbl.grid(row=1, column=0, padx=8, pady=(0, 4), sticky="w")

            status = ctk.CTkLabel(
                card,
                text="3 Support | D=48",
                font=ctk.CTkFont(size=9, weight="bold"),
                text_color="#10b981",
                fg_color="#064e3b",
                corner_radius=4,
                padx=6,
                pady=1
            )
            status.grid(row=0, column=1, rowspan=2, padx=8, pady=4, sticky="e")

    def _on_toggle_exec(self):
        self.execute_real = bool(self.switch_exec.get())
        if self.execute_real:
            self.lbl_action_status.configure(
                text="⚠️ Chế độ THỰC THI THẬT đã bật: Ứng dụng sẽ mở trực tiếp trên Windows khi nhận diện!",
                text_color="#f59e0b"
            )
        else:
            self.lbl_action_status.configure(
                text="ℹ️ Chế độ Giả lập an toàn: Chỉ ghi nhận kết quả, không mở ứng dụng thật.",
                text_color="#94a3b8"
            )

    def _on_click_reset_samples(self):
        ensure_samples()
        self.engine.enroll_all()
        self._render_keyword_cards()
        self.status_badge.configure(text="● ĐÃ NẠP LẠI MẪU CHUẨN THÀNH CÔNG", text_color="#10b981")

    def _on_click_test_sample(self, filename: str):
        qp = os.path.join(SAMPLES_DIR, filename)
        self._process_query(qp, display_name=filename)

    def _on_click_live_mic(self):
        """Records 1 second from real microphone in background thread."""
        def record_thread():
            try:
                import sounddevice as sd
            except ImportError:
                self.lbl_decision.configure(text="LỖI: CHƯA CÀI SOUNDDEVICE", text_color="#ef4444")
                return

            self.btn_mic.configure(state="disabled", text="🔴 ĐANG THU ÂM... NÓI NGAY BÂY GIỜ!", fg_color="#ef4444")
            self.lbl_decision.configure(text="ĐANG LẮNG NGHE...", text_color="#f59e0b")

            sr = 16000
            duration_s = 1.0
            audio = sd.rec(int(duration_s * sr), samplerate=sr, channels=1, dtype='int16')
            sd.wait()

            query_fp = os.path.join(SAMPLES_DIR, "query_live_mic.wav")
            with wave.open(query_fp, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio.tobytes())

            self.btn_mic.configure(state="normal", text="🎙️ BẤM VÀ NÓI VÀO MICRO (1S)", fg_color="#8b5cf6")
            self._process_query(query_fp, display_name="Microphone (Giọng thật)")

        threading.Thread(target=record_thread, daemon=True).start()

    def _on_click_enroll_live(self):
        """Dialog to record custom voice for a selected keyword."""
        dialog = ctk.CTkInputDialog(
            text="Nhập tên từ khóa muốn thu âm giọng thật:\n(open_notepad / open_browser / system_mute / stop_task)",
            title="Thu âm giọng thật (3-shot)"
        )
        kw = dialog.get_input()
        if not kw or kw not in KEYWORDS:
            return

        def enroll_thread():
            import sounddevice as sd
            kw_dir = os.path.join(SUPPORT_DIR, kw)
            os.makedirs(kw_dir, exist_ok=True)

            for shot in range(1, 4):
                self.status_badge.configure(
                    text=f"🔴 NÓI TỪ '{kw}' (MẪU {shot}/3)...",
                    text_color="#f59e0b",
                    fg_color="#78350f"
                )
                time.sleep(1.0)
                audio = sd.rec(16000, samplerate=16000, channels=1, dtype='int16')
                sd.wait()

                fp = os.path.join(kw_dir, f"sample_{shot}.wav")
                with wave.open(fp, "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio.tobytes())
                time.sleep(0.5)

            self.engine.enroll_all()
            self._render_keyword_cards()
            self.status_badge.configure(
                text=f"● ĐÃ LƯU GIỌNG THẬT CHO '{kw}'",
                text_color="#10b981",
                fg_color="#064e3b"
            )

        threading.Thread(target=enroll_thread, daemon=True).start()

    def _process_query(self, wav_path: str, display_name: str = ""):
        res = self.engine.classify(wav_path)
        pred = res["predicted"]
        conf = res["confidence"]
        lat = res["latency_ms"]
        dists = res["distances"]
        is_rej = res["is_rejected"]

        # 1. Update Decision Text
        if is_rej:
            self.lbl_decision.configure(
                text="⚠️ TỪ CHỐI (TẠP ÂM / TỪ NGOÀI TỪ ĐIỂN)",
                text_color="#ef4444"
            )
            self.lbl_action_status.configure(
                text="[REJECTED] Tín hiệu bị loại bỏ do độ tin cậy thấp hoặc khoảng cách quá lớn -> Tránh kích hoạt nhầm.",
                text_color="#ef4444"
            )
        else:
            meta = KEYWORDS.get(pred, {})
            self.lbl_decision.configure(
                text=f"{meta.get('icon', '✓')} {pred.upper()}",
                text_color="#10b981"
            )
            action_desc = meta.get('action', '')
            cmd = meta.get('cmd')

            if self.execute_real and cmd:
                try:
                    subprocess.Popen(cmd, shell=True)
                    self.lbl_action_status.configure(
                        text=f"✓ ĐÃ THỰC THI THẬT TRÊN WINDOWS: {action_desc} ({cmd})",
                        text_color="#10b981"
                    )
                except Exception as e:
                    self.lbl_action_status.configure(text=f"Lỗi thực thi: {e}", text_color="#ef4444")
            else:
                self.lbl_action_status.configure(
                    text=f"✓ Nhận diện thành công: {action_desc} (Chế độ mô phỏng an toàn)",
                    text_color="#38bdf8"
                )

        # 2. Update Confidence & Latency
        self.lbl_conf.configure(text=f"Độ tin cậy: {conf:.1f}%")
        self.lbl_latency.configure(text=f"⚡ Độ trễ PyTorch: {lat:.2f} ms")
        self.progress_conf.set(conf / 100.0)
        self.progress_conf.configure(progress_color="#10b981" if not is_rej else "#ef4444")

        # 3. Update Distance Bars
        best_k = min(dists, key=dists.get)
        for k in KEYWORDS:
            d = dists.get(k, 1.5)
            # Normalize bar progress: lower distance means higher match (bar fuller)
            # d in [0, 1.5] -> score in [1, 0]
            score = max(0.0, min(1.0, 1.0 - (d / 1.5)))
            self.dist_bars[k].set(score)
            if k == best_k and not is_rej:
                self.dist_bars[k].configure(progress_color="#10b981")
                self.dist_labels[k].configure(text=f"d = {d:.4f} ★", text_color="#10b981")
            else:
                self.dist_bars[k].configure(progress_color="#3b82f6" if d < 1.0 else "#64748b")
                self.dist_labels[k].configure(text=f"d = {d:.4f}", text_color="#94a3b8")


def main():
    app = VoiceAssistantApp()
    app.mainloop()


if __name__ == "__main__":
    main()
