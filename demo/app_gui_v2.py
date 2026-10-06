"""Voice — minimal desktop assistant UI (V2)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Allow this module to run both as ``python -m demo.app_gui_v2`` and from the
# Windows launcher via ``python demo\app_gui_v2.py``.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import customtkinter as ctk
import numpy as np

from demo.antigravity_runner import (
    AntigravityActionResolver,
    AntigravityChatSession,
    CachedActionExecutor,
)
from demo.app_gui import (
    FewShotEngine,
    KEYWORDS_DIR,
    METADATA_FILE,
    PROJECT_ROOT,
    calculate_rms,
    normalize_and_save_wav,
    sd,
    slugify,
)
from src.audio_dsp import SileroVADManager


try:
    import winsound
except ImportError:
    winsound = None

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG = ("#f5f5f7", "#000000")
SURFACE = ("#ffffff", "#161617")
SURFACE_2 = ("#f0f0f2", "#232325")
TEXT = ("#1d1d1f", "#f5f5f7")
MUTED = ("#6e6e73", "#a1a1a6")
ACCENT = "#0a84ff"
SUCCESS = "#30d158"
WARNING = "#ff9f0a"
PURPLE = "#bf5af2"
SETTINGS_FILE = os.path.join(KEYWORDS_DIR, "settings.json")
RUNTIME_LOG = os.path.join(PROJECT_ROOT, "demo", "logs", "voice_runtime.log")
OPEN_UI_SIGNAL = os.path.join(PROJECT_ROOT, "demo", "logs", "open_ui.signal")
ORB_STATUS_FILE = os.path.join(PROJECT_ROOT, "demo", "logs", "orb_status.json")
SHUTDOWN_SIGNAL = os.path.join(PROJECT_ROOT, "demo", "logs", "shutdown.signal")

SOUND_DIR = os.path.join(PROJECT_ROOT, "demo", "assets", "sounds")
SOUND_CLICK = os.path.join(SOUND_DIR, "click.wav")
SOUND_CONFIRM = os.path.join(SOUND_DIR, "confirm.wav")
SOUND_GAMING_LOCK = os.path.join(SOUND_DIR, "gaming_lock.wav")


def play_sound(sound_path: str) -> None:
    """Phát âm thanh phản hồi UI trên luồng nền daemon không gây gián đoạn giao diện."""
    if winsound is None or not os.path.exists(sound_path):
        return

    def _worker():
        try:
            winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(settings):
    os.makedirs(KEYWORDS_DIR, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)


class CommandSheet(ctk.CTkToplevel):
    def __init__(self, parent, on_saved):
        super().__init__(parent)
        self.on_saved = on_saved
        self.samples = []
        self.recording = False
        self.title("New Command")
        self.geometry("520x590")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.configure(fg_color=BG)
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="New voice command", font=ctk.CTkFont(size=26, weight="bold"), text_color=TEXT).pack(anchor="w", padx=30, pady=(28, 4))
        ctk.CTkLabel(self, text="Five samples minimum. Add as many as you want for better accuracy.", font=ctk.CTkFont(size=13), text_color=MUTED).pack(anchor="w", padx=30, pady=(0, 22))

        card = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=20)
        card.pack(fill="both", expand=True, padx=24, pady=(0, 18))

        self.name_entry = ctk.CTkEntry(card, placeholder_text="Command name", height=46, corner_radius=12, border_width=0, fg_color=SURFACE_2, font=ctk.CTkFont(size=14))
        self.name_entry.pack(fill="x", padx=18, pady=(20, 10))
        self.desc_entry = ctk.CTkTextbox(card, height=105, corner_radius=12, border_width=0, fg_color=SURFACE_2, font=ctk.CTkFont(size=13), wrap="word")
        self.desc_entry.pack(fill="x", padx=18, pady=(0, 16))
        self.desc_entry.insert("1.0", "Describe what Antigravity should do")

        self.progress = ctk.CTkProgressBar(card, height=5, progress_color=ACCENT)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=24, pady=(12, 8))
        self.status = ctk.CTkLabel(card, text="0 of 5 recordings", font=ctk.CTkFont(size=12), text_color=MUTED)
        self.status.pack(pady=(0, 12))
        self.record_button = ctk.CTkButton(card, text="Record sample", command=self._record, height=48, corner_radius=24, fg_color=TEXT, hover_color=MUTED, text_color=BG, font=ctk.CTkFont(size=14, weight="bold"))
        self.record_button.pack(padx=24, pady=(0, 20), fill="x")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=30, pady=(0, 24))
        ctk.CTkButton(actions, text="Cancel", command=self.destroy, width=100, fg_color="transparent", hover_color=SURFACE_2, text_color=TEXT).pack(side="left")
        self.save_button = ctk.CTkButton(actions, text="Save", command=self._save, width=120, state="disabled", corner_radius=18, fg_color=ACCENT)
        self.save_button.pack(side="right")

    def _record(self):
        if self.recording or sd is None:
            return
        if not self.name_entry.get().strip():
            self.status.configure(text="Enter a command name first", text_color=WARNING)
            return
        self.recording = True
        self.record_button.configure(state="disabled", text="Listening…")

        def worker():
            audio = sd.rec(int(1.3 * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            self.samples.append(audio.flatten())
            self.after(0, self._recorded)

        threading.Thread(target=worker, daemon=True).start()

    def _recorded(self):
        self.recording = False
        count = len(self.samples)
        self.progress.set(min(1.0, count / 5))
        if count >= 5:
            self.status.configure(text=f"{count} recordings · minimum reached", text_color=SUCCESS)
            self.record_button.configure(text="Add another sample", state="normal")
            self.save_button.configure(state="normal")
        else:
            self.status.configure(text=f"{count} of 5 minimum recordings", text_color=MUTED)
            self.record_button.configure(text="Record next sample", state="normal")

    def _save(self):
        name = self.name_entry.get().strip()
        description = self.desc_entry.get("1.0", "end").strip()
        if not name or not description or description == "Describe what Antigravity should do" or len(self.samples) < 5:
            self.status.configure(text="Name, action and at least five samples are required", text_color=WARNING)
            return
        slug = slugify(name)
        target = os.path.join(KEYWORDS_DIR, slug)
        os.makedirs(target, exist_ok=True)
        for index, sample in enumerate(self.samples, 1):
            normalize_and_save_wav(sample, os.path.join(target, f"sample_{index}.wav"))
        metadata = load_metadata()
        metadata[slug] = {
            "name": name,
            "description": description,
            "action_type": "antigravity",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "samples_count": len(self.samples),
        }
        with open(METADATA_FILE, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        self.on_saved()
        self.destroy()


class RecordMoreSheet(ctk.CTkToplevel):
    def __init__(self, parent, slug, item, on_saved):
        super().__init__(parent)
        self.slug = slug
        self.item = item
        self.on_saved = on_saved
        self.cmd_name = item.get("name", slug)
        self.new_samples = []
        self.recording = False

        target_dir = os.path.join(KEYWORDS_DIR, self.slug)
        existing_wavs = [f for f in os.listdir(target_dir) if f.endswith(".wav")] if os.path.isdir(target_dir) else []
        self.existing_count = len(existing_wavs)

        self.title(f"Add Samples · {self.cmd_name}")
        self.geometry("520x490")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.configure(fg_color=BG)
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text=f"Record more: {self.cmd_name}", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT).pack(anchor="w", padx=30, pady=(26, 4))
        ctk.CTkLabel(self, text="Add extra voice recordings to make this command even more accurate.", font=ctk.CTkFont(size=13), text_color=MUTED).pack(anchor="w", padx=30, pady=(0, 20))

        card = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=20)
        card.pack(fill="both", expand=True, padx=24, pady=(0, 18))

        stats_frame = ctk.CTkFrame(card, fg_color=SURFACE_2, corner_radius=12)
        stats_frame.pack(fill="x", padx=18, pady=(18, 14))
        self.stats_label = ctk.CTkLabel(
            stats_frame,
            text=f"Current: {self.existing_count} samples  ·  New: 0 added",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT,
        )
        self.stats_label.pack(pady=12)

        self.progress = ctk.CTkProgressBar(card, height=5, progress_color=ACCENT)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=24, pady=(8, 8))

        self.status = ctk.CTkLabel(card, text="Ready to record new sample", font=ctk.CTkFont(size=13), text_color=MUTED)
        self.status.pack(pady=(0, 14))

        self.record_button = ctk.CTkButton(
            card,
            text="Record sample",
            command=self._record,
            height=48,
            corner_radius=24,
            fg_color=TEXT,
            hover_color=MUTED,
            text_color=BG,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.record_button.pack(padx=24, pady=(0, 18), fill="x")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=30, pady=(0, 22))
        ctk.CTkButton(actions, text="Cancel", command=self.destroy, width=100, fg_color="transparent", hover_color=SURFACE_2, text_color=TEXT).pack(side="left")
        self.save_button = ctk.CTkButton(actions, text="Save (0)", command=self._save, width=130, state="disabled", corner_radius=18, fg_color=ACCENT)
        self.save_button.pack(side="right")

    def _record(self):
        if self.recording or sd is None:
            return
        self.recording = True
        self.record_button.configure(state="disabled", text="Listening…")
        self.status.configure(text="Speak now…", text_color=ACCENT)

        def worker():
            audio = sd.rec(int(1.3 * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            self.new_samples.append(audio.flatten())
            self.after(0, self._recorded)

        threading.Thread(target=worker, daemon=True).start()

    def _recorded(self):
        self.recording = False
        count = len(self.new_samples)
        self.progress.set(min(1.0, count / 5))
        self.stats_label.configure(text=f"Current: {self.existing_count} samples  ·  New: +{count} added (Total: {self.existing_count + count})")
        self.status.configure(text=f"+{count} new sample(s) recorded · Ready to save or add more", text_color=SUCCESS)
        self.record_button.configure(text="Add another sample", state="normal")
        self.save_button.configure(state="normal", text=f"Save (+{count})")

    def _save(self):
        if not self.new_samples:
            return
        target = os.path.join(KEYWORDS_DIR, self.slug)
        os.makedirs(target, exist_ok=True)
        import re
        existing_indices = []
        for fname in os.listdir(target):
            m = re.match(r"^sample_(\d+)\.wav$", fname)
            if m:
                existing_indices.append(int(m.group(1)))
        next_idx = max(existing_indices, default=0) + 1

        for sample in self.new_samples:
            normalize_and_save_wav(sample, os.path.join(target, f"sample_{next_idx}.wav"))
            next_idx += 1

        total_wavs = len([f for f in os.listdir(target) if f.endswith(".wav")])
        metadata = load_metadata()
        if self.slug not in metadata:
            metadata[self.slug] = self.item
        metadata[self.slug]["samples_count"] = total_wavs
        with open(METADATA_FILE, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        self.on_saved()
        self.destroy()


class EditCommandSheet(ctk.CTkToplevel):
    def __init__(self, parent, slug, item, on_saved):
        super().__init__(parent)
        self.slug = slug
        self.item = item
        self.on_saved = on_saved
        self.cmd_name = item.get("name", slug)
        self.old_desc = item.get("description", "").strip()

        self.title(f"Chỉnh sửa lệnh · {self.cmd_name}")
        self.geometry("520x460")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.configure(fg_color=BG)
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text=f"Chỉnh sửa: {self.cmd_name}", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT).pack(anchor="w", padx=30, pady=(26, 4))
        ctk.CTkLabel(self, text="Cập nhật yêu cầu thực thi cho Antigravity AI.", font=ctk.CTkFont(size=13), text_color=MUTED).pack(anchor="w", padx=30, pady=(0, 18))

        card = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=20)
        card.pack(fill="both", expand=True, padx=24, pady=(0, 18))

        ctk.CTkLabel(card, text="Mô tả hành động (Prompt Antigravity):", font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT).pack(anchor="w", padx=20, pady=(16, 6))

        self.desc_entry = ctk.CTkTextbox(card, height=140, corner_radius=12, border_width=0, fg_color=SURFACE_2, font=ctk.CTkFont(size=13), wrap="word")
        self.desc_entry.pack(fill="x", padx=18, pady=(0, 10))
        self.desc_entry.insert("1.0", self.old_desc)

        has_profile = bool(self.item.get("action_profile"))
        profile_hint = "ℹ️ Lệnh đã có code cache đã học. Nếu sửa nội dung khác đi, hệ thống sẽ xóa cache cũ để Antigravity tự động học lại!" if has_profile else "ℹ️ Lệnh chưa có code cache đã học."
        self.status = ctk.CTkLabel(card, text=profile_hint, font=ctk.CTkFont(size=11), text_color=WARNING if has_profile else MUTED, wraplength=440, justify="left")
        self.status.pack(anchor="w", padx=20, pady=(0, 12))

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=30, pady=(0, 22))
        ctk.CTkButton(actions, text="Hủy", command=self.destroy, width=90, fg_color="transparent", hover_color=SURFACE_2, text_color=TEXT).pack(side="left")
        self.save_button = ctk.CTkButton(actions, text="Lưu thay đổi", command=self._save, width=130, corner_radius=18, fg_color=ACCENT)
        self.save_button.pack(side="right")

    def _save(self):
        new_desc = self.desc_entry.get("1.0", "end").strip()
        if not new_desc:
            self.status.configure(text="⚠️ Mô tả hành động không được để trống!", text_color=WARNING)
            return

        metadata = load_metadata()
        if self.slug not in metadata:
            metadata[self.slug] = self.item

        # SO SÁNH VỚI MÔ TẢ TRƯỚC ĐÓ ĐƯỢC ANTIGRAVITY THỰC THI
        if new_desc != self.old_desc:
            metadata[self.slug]["description"] = new_desc
            # Xóa action_profile cũ đã cache để bắt buộc Antigravity học lại từ đầu
            if "action_profile" in metadata[self.slug]:
                del metadata[self.slug]["action_profile"]
            metadata[self.slug]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(METADATA_FILE, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        play_sound(SOUND_CONFIRM)
        self.on_saved()
        self.destroy()


class VoiceAppV2(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Voice")
        self.geometry("820x600")
        self.minsize(740, 540)
        self.configure(fg_color=BG)
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self.engine = FewShotEngine()
        self.stream = None
        self.buffer = np.zeros(int(1.3 * 16000), dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.listening = True
        self.last_detection = 0.0
        self.action_running = False
        self.action_last_started = {}
        self.action_executor = CachedActionExecutor()
        self.wake_slug = load_settings().get("wake_word", "")
        self.awake_until = 0.0
        self.wake_window_seconds = 10
        self.wake_timer_id = None
        self.wake_countdown_id = None
        self.chat = None
        self.chat_busy = False
        self.chat_reply = None
        self.compact_mode = False
        self.orb_process = None
        self.is_calibrating = False

        self._build_shell()
        self._show_page("voice")
        self._refresh_commands()
        self._start_audio()
        self.after(600, self._auto_calibrate_noise)
        if "--daemon" in sys.argv or "--compact" in sys.argv:
            self.after(900, self._collapse)
        self.after(250, self._poll_open_ui_signal)

    def _log(self, event, **data):
        try:
            os.makedirs(os.path.dirname(RUNTIME_LOG), exist_ok=True)
            record = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **data}
            with open(RUNTIME_LOG, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _set_orb_status(self, title, detail):
        try:
            os.makedirs(os.path.dirname(ORB_STATUS_FILE), exist_ok=True)
            with open(ORB_STATUS_FILE, "w", encoding="utf-8") as handle:
                json.dump({"title": title, "detail": detail}, handle, ensure_ascii=False)
        except OSError:
            pass

    def _build_shell(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.rail = ctk.CTkFrame(self, width=164, corner_radius=0, fg_color=SURFACE)
        self.rail.grid(row=0, column=0, sticky="nsew")
        self.rail.grid_rowconfigure(5, weight=1)
        ctk.CTkLabel(self.rail, text="Voice", font=ctk.CTkFont(size=23, weight="bold"), text_color=TEXT).grid(row=0, column=0, padx=20, pady=(24, 22), sticky="w")

        self.nav_buttons = {}
        for row, (key, label) in enumerate((("voice", "Voice"), ("commands", "Commands"), ("chat", "Chat")), 1):
            button = ctk.CTkButton(self.rail, text=label, anchor="w", height=40, corner_radius=12, command=lambda page=key: self._show_page(page), fg_color="transparent", hover_color=SURFACE_2, text_color=TEXT, font=ctk.CTkFont(size=13, weight="bold"))
            button.grid(row=row, column=0, padx=14, pady=3, sticky="ew")
            self.nav_buttons[key] = button

        self.connection = ctk.CTkLabel(self.rail, text="●  Ready", font=ctk.CTkFont(size=11), text_color=SUCCESS)
        self.connection.grid(row=6, column=0, padx=24, pady=(8, 6), sticky="w")
        self.theme_button = ctk.CTkButton(self.rail, text="Appearance", command=self._toggle_theme, height=34, fg_color="transparent", hover_color=SURFACE_2, text_color=MUTED)
        self.theme_button.grid(row=7, column=0, padx=14, pady=(0, 18), sticky="ew")
        ctk.CTkButton(self.rail, text="Minimize", command=self._collapse, height=34, fg_color="transparent", hover_color=SURFACE_2, text_color=MUTED).grid(row=8, column=0, padx=14, pady=(0, 6), sticky="ew")
        ctk.CTkButton(self.rail, text="Quit", command=self._quit, height=30, fg_color="transparent", hover_color=SURFACE_2, text_color=MUTED).grid(row=9, column=0, padx=14, pady=(0, 14), sticky="ew")

        self.content = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.pages = {
            "voice": self._build_voice_page(),
            "commands": self._build_commands_page(),
            "chat": self._build_chat_page(),
        }

    def _collapse(self):
        self.compact_mode = True
        # Đóng dứt điểm tiến trình orb overlay nếu đang mở
        if self.orb_process is not None:
            try:
                if self.orb_process.poll() is None:
                    self.orb_process.terminate()
            except Exception:
                pass
            self.orb_process = None
        # Dọn sạch các file tín hiệu trung gian
        for signal in (OPEN_UI_SIGNAL, SHUTDOWN_SIGNAL):
            try:
                if os.path.exists(signal):
                    os.remove(signal)
            except OSError:
                pass
        self.rail.grid_remove()
        self.content.grid_remove()
        self.overrideredirect(True)
        self.withdraw()
        self._update_wake_status()

    def _show_wake_orb(self):
        # NGUYÊN TẮC ĐỘC QUYỀN (MUTUALLY EXCLUSIVE):
        # Floating Orb Overlay CHỈ ĐƯỢC PHÉP XUẤT HIỆN khi cửa sổ chính (Panel) đang thu nhỏ (compact_mode = True).
        # Khi người dùng đang mở bảng điều khiển chính, tuyệt đối không bao giờ khởi chạy Orb!
        if not self.compact_mode or self.state() != "withdrawn":
            return
        overlay = os.path.join(PROJECT_ROOT, "demo", "orb_overlay.py")
        try:
            for signal in (OPEN_UI_SIGNAL, SHUTDOWN_SIGNAL):
                if os.path.exists(signal):
                    os.remove(signal)
            self._set_orb_status("Đang nghe", "Hãy nói một command")
            if self.orb_process is None or self.orb_process.poll() is not None:
                self.orb_process = subprocess.Popen(
                    [sys.executable, overlay, OPEN_UI_SIGNAL, ORB_STATUS_FILE, SHUTDOWN_SIGNAL, str(self.wake_window_seconds)],
                    cwd=PROJECT_ROOT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                self._log("orb_started", pid=self.orb_process.pid)
        except Exception as exc:
            self._log("orb_error", error=str(exc))

    def _poll_open_ui_signal(self):
        if os.path.exists(SHUTDOWN_SIGNAL):
            try:
                os.remove(SHUTDOWN_SIGNAL)
            except OSError:
                pass
            # Chỉ tắt toàn bộ ứng dụng nếu người dùng bấm [x] trên Orb trong lúc đang ở compact mode
            if self.compact_mode:
                self._log("shutdown_from_orb")
                self._quit()
                return
        if os.path.exists(OPEN_UI_SIGNAL):
            try:
                os.remove(OPEN_UI_SIGNAL)
            except OSError:
                pass
            self._log("orb_clicked")
            self._expand(from_wake=True)
        if self.winfo_exists():
            self.after(250, self._poll_open_ui_signal)

    def _expand(self, from_wake=False):
        self.compact_mode = False
        # Đảm bảo tắt hẳn tiến trình orb trước khi khôi phục bảng điều khiển chính
        if self.orb_process is not None:
            try:
                if self.orb_process.poll() is None:
                    self.orb_process.terminate()
            except Exception:
                pass
            self.orb_process = None
        for signal in (OPEN_UI_SIGNAL, SHUTDOWN_SIGNAL):
            try:
                if os.path.exists(signal):
                    os.remove(signal)
            except OSError:
                pass
        self.overrideredirect(False)
        self.resizable(True, True)
        width, height = 820, 600
        x = max(0, (self.winfo_screenwidth() - width) // 2)
        y = max(0, (self.winfo_screenheight() - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.rail.grid()
        self.content.grid()
        self.deiconify()
        self.lift()
        if from_wake:
            self.focus_force()
        self._update_wake_status()
        self.after(1200, lambda: self.attributes("-topmost", False))

    def _quit(self):
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self._close()

    def _page(self):
        frame = ctk.CTkFrame(self.content, fg_color=BG, corner_radius=0)
        frame.grid(row=0, column=0, sticky="nsew")
        return frame

    def _build_voice_page(self):
        page = self._page()
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(page, text="Listening", font=ctk.CTkFont(size=15, weight="bold"), text_color=MUTED).grid(row=0, column=0, padx=40, pady=(34, 0), sticky="w")
        center = ctk.CTkFrame(page, fg_color="transparent")
        center.grid(row=1, column=0)
        self.orb = ctk.CTkButton(center, text="", width=190, height=190, corner_radius=95, command=self._toggle_listening, fg_color=TEXT, hover_color=MUTED, border_width=8, border_color=SURFACE_2)
        self.orb.pack(pady=(0, 28))
        self.voice_title = ctk.CTkLabel(center, text="Say a command", font=ctk.CTkFont(size=30, weight="bold"), text_color=TEXT)
        self.voice_title.pack()
        self.voice_detail = ctk.CTkLabel(center, text="Your assistant is ready", font=ctk.CTkFont(size=14), text_color=MUTED)
        self.voice_detail.pack(pady=(8, 22))
        self.voice_meter = ctk.CTkProgressBar(center, width=300, height=4, progress_color=ACCENT, fg_color=SURFACE_2)
        self.voice_meter.set(0)
        self.voice_meter.pack()
        self.calibrate_button = ctk.CTkButton(
            center,
            text="🎯 Đo tiếng ồn phòng (1s)",
            command=self._manual_calibrate_noise,
            height=34,
            corner_radius=17,
            fg_color=SURFACE_2,
            hover_color=MUTED,
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.calibrate_button.pack(pady=(20, 6))
        self.noise_info = ctk.CTkLabel(
            center,
            text=f"Ồn phòng: {self.engine.ambient_noise_rms:.3f} · Bắt giọng: >{self.engine.speech_trigger_rms:.3f}",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
        )
        self.noise_info.pack()
        return page

    def _build_commands_page(self):
        page = self._page()
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, padx=36, pady=(30, 18), sticky="ew")
        ctk.CTkLabel(header, text="Commands", font=ctk.CTkFont(size=28, weight="bold"), text_color=TEXT).pack(side="left")
        ctk.CTkButton(header, text="New", width=78, height=36, corner_radius=18, command=lambda: CommandSheet(self, self._refresh_commands), fg_color=ACCENT).pack(side="right")
        self.wake_menu = ctk.CTkOptionMenu(header, values=["Wake word: Off"], command=self._select_wake_word, width=180, height=36, corner_radius=18, fg_color=SURFACE_2, button_color=SURFACE_2, button_hover_color=MUTED, text_color=TEXT)
        self.wake_menu.pack(side="right", padx=10)
        self.command_list = ctk.CTkScrollableFrame(page, fg_color="transparent")
        self.command_list.grid(row=1, column=0, padx=30, pady=(0, 28), sticky="nsew")
        return page

    def _build_chat_page(self):
        page = self._page()
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, padx=36, pady=(30, 14), sticky="ew")
        ctk.CTkLabel(header, text="Antigravity", font=ctk.CTkFont(size=28, weight="bold"), text_color=TEXT).pack(side="left")
        self.tool_switch = ctk.CTkSwitch(header, text="Actions", font=ctk.CTkFont(size=12), progress_color=ACCENT)
        self.tool_switch.pack(side="right")
        self.chat_status = ctk.CTkLabel(header, text="Persistent stream", font=ctk.CTkFont(size=11), text_color=MUTED)
        self.chat_status.pack(side="right", padx=14)

        self.messages = ctk.CTkScrollableFrame(page, fg_color="transparent")
        self.messages.grid(row=1, column=0, padx=30, sticky="nsew")
        self.messages.grid_columnconfigure(0, weight=1)
        self._bubble("assistant", "How can I help?")

        composer = ctk.CTkFrame(page, fg_color=SURFACE, corner_radius=22)
        composer.grid(row=2, column=0, padx=36, pady=26, sticky="ew")
        composer.grid_columnconfigure(0, weight=1)
        self.chat_input = ctk.CTkTextbox(composer, height=62, fg_color="transparent", border_width=0, font=ctk.CTkFont(size=14), wrap="word")
        self.chat_input.grid(row=0, column=0, padx=(14, 6), pady=8, sticky="ew")

        def _on_chat_return(event):
            if not (event.state & 0x0001):  # not Shift+Enter
                self._send_chat()
                return "break"

        self.chat_input.bind("<Return>", _on_chat_return)
        self.chat_input.bind("<Control-Return>", lambda _event: self._send_chat())
        self.send_button = ctk.CTkButton(composer, text="↑", width=42, height=42, corner_radius=21, command=self._send_chat, fg_color=TEXT, hover_color=MUTED, text_color=BG, font=ctk.CTkFont(size=18, weight="bold"))
        self.send_button.grid(row=0, column=1, padx=(0, 12), pady=12)
        return page

    def _show_page(self, name):
        for key, page in self.pages.items():
            if key == name:
                page.tkraise()
                self.nav_buttons[key].configure(fg_color=SURFACE_2)
            else:
                self.nav_buttons[key].configure(fg_color="transparent")

    def _toggle_theme(self):
        ctk.set_appearance_mode("light" if ctk.get_appearance_mode().lower() == "dark" else "dark")

    def _refresh_commands(self):
        for child in self.command_list.winfo_children():
            child.destroy()
        self.engine.update_prototypes()
        metadata = load_metadata()
        self._wake_choices = {"Wake word: Off": ""}
        for slug, item in metadata.items():
            self._wake_choices[f"Wake word: {item.get('name', slug)}"] = slug
        self.wake_menu.configure(values=list(self._wake_choices))
        selected = next((label for label, slug in self._wake_choices.items() if slug == self.wake_slug), "Wake word: Off")
        self.wake_menu.set(selected)
        if self.wake_slug not in metadata:
            self.wake_slug = ""
            save_settings({"wake_word": ""})
        self._update_wake_status()
        if not metadata:
            ctk.CTkLabel(self.command_list, text="No commands yet", text_color=MUTED, font=ctk.CTkFont(size=14)).pack(pady=80)
            return
        for slug, item in metadata.items():
            row = ctk.CTkFrame(self.command_list, fg_color=SURFACE, corner_radius=16)
            row.pack(fill="x", padx=4, pady=6)
            row.grid_columnconfigure(0, weight=1)
            row.grid_columnconfigure(1, weight=0)

            text = ctk.CTkFrame(row, fg_color="transparent")
            text.grid(row=0, column=0, sticky="nsew", padx=(18, 12), pady=14)

            title_row = ctk.CTkFrame(text, fg_color="transparent")
            title_row.pack(anchor="w")
            ctk.CTkLabel(title_row, text=item.get("name", slug), font=ctk.CTkFont(size=15, weight="bold"), text_color=TEXT).pack(side="left")
            samples_count = item.get("samples_count")
            if samples_count is None:
                kw_dir = os.path.join(KEYWORDS_DIR, slug)
                samples_count = len([f for f in os.listdir(kw_dir) if f.endswith(".wav")]) if os.path.isdir(kw_dir) else 0
            badge = ctk.CTkLabel(title_row, text=f"·  {samples_count} samples", font=ctk.CTkFont(size=12), text_color=MUTED)
            badge.pack(side="left", padx=8)

            desc = item.get("description", "")
            desc_label = ctk.CTkLabel(text, text=desc, font=ctk.CTkFont(size=12), text_color=MUTED, wraplength=380, justify="left")
            desc_label.pack(anchor="w", pady=(3, 0))

            btn_frame = ctk.CTkFrame(row, fg_color="transparent")
            btn_frame.grid(row=0, column=1, sticky="e", padx=(0, 16), pady=14)
            ctk.CTkButton(
                btn_frame,
                text="Sửa",
                width=52,
                height=32,
                corner_radius=14,
                fg_color=SURFACE_2,
                hover_color=MUTED,
                text_color=TEXT,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda key=slug, it=item: EditCommandSheet(self, key, it, self._refresh_commands),
            ).pack(side="left", padx=(0, 6))
            ctk.CTkButton(
                btn_frame,
                text="+ Mẫu",
                width=64,
                height=32,
                corner_radius=14,
                fg_color=SURFACE_2,
                hover_color=MUTED,
                text_color=TEXT,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda key=slug, it=item: RecordMoreSheet(self, key, it, self._refresh_commands),
            ).pack(side="left", padx=(0, 6))
            ctk.CTkButton(
                btn_frame,
                text="Xóa",
                width=56,
                height=32,
                corner_radius=14,
                fg_color="transparent",
                hover_color=SURFACE_2,
                text_color=MUTED,
                font=ctk.CTkFont(size=12),
                command=lambda key=slug: self._remove_command(key),
            ).pack(side="left")

    def _select_wake_word(self, label):
        self.wake_slug = self._wake_choices.get(label, "")
        self.awake_until = 0.0
        save_settings({"wake_word": self.wake_slug})
        self._update_wake_status()

    def _update_wake_status(self):
        if not hasattr(self, "voice_title"):
            return
        if hasattr(self, "orb"):
            self.orb.configure(border_color=SURFACE_2)
        if self.wake_slug:
            item = load_metadata().get(self.wake_slug, {})
            name = item.get("name", self.wake_slug)
            self.voice_title.configure(text=f'Say “{name}”')
            self.voice_detail.configure(text="Sleeping · waiting for wake word", text_color=MUTED)
            self.connection.configure(text="●  Sleeping", text_color=MUTED)
        else:
            self.voice_title.configure(text="Say a command")
            self.voice_detail.configure(text="Your assistant is ready", text_color=MUTED)
            self.connection.configure(text="●  Ready", text_color=SUCCESS)

    def _remove_command(self, slug):
        target = os.path.join(KEYWORDS_DIR, slug)
        if os.path.isdir(target):
            shutil.rmtree(target)
        metadata = load_metadata()
        metadata.pop(slug, None)
        with open(METADATA_FILE, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        if self.wake_slug == slug:
            self.wake_slug = ""
            self.awake_until = 0.0
            save_settings({"wake_word": ""})
        self._refresh_commands()

    def _start_audio(self):
        if sd is None:
            self.connection.configure(text="●  Microphone unavailable", text_color=WARNING)
            return
        def callback(indata, _frames, _time_info, _status):
            if not self.listening:
                return
            chunk = indata[:, 0]
            with self.buffer_lock:
                self.buffer = np.roll(self.buffer, -len(chunk))
                self.buffer[-len(chunk):] = chunk
        try:
            self.stream = sd.InputStream(samplerate=16000, channels=1, blocksize=1024, dtype="float32", callback=callback)
            self.stream.start()
            self._log("microphone_started", device=str(sd.default.device), wake_word=self.wake_slug)
            threading.Thread(target=self._listen_loop, daemon=True).start()
        except Exception as exc:
            self._log("microphone_error", error=str(exc))
            self.connection.configure(text=f"●  {exc}", text_color=WARNING)

    def _auto_calibrate_noise(self):
        if sd is None or self.is_calibrating:
            return
        threading.Thread(target=self._calibrate_worker, daemon=True).start()

    def _manual_calibrate_noise(self):
        if sd is None or self.is_calibrating:
            return
        threading.Thread(target=self._calibrate_worker, daemon=True).start()

    def _calibrate_worker(self):
        self.is_calibrating = True
        self.after(0, lambda: self.calibrate_button.configure(text="⏳ Đang đo ồn phòng...", state="disabled"))
        self.after(0, lambda: self.voice_detail.configure(text="🎯 Giữ im lặng 1 giây để hệ thống đo tiếng ồn phòng...", text_color=ACCENT))
        try:
            rec = sd.rec(int(1.0 * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            noise_audio = rec.flatten()
            self.engine.calibrate_ambient_noise(noise_audio)
            self._log(
                "noise_calibrated",
                ambient_rms=round(float(self.engine.ambient_noise_rms), 4),
                speech_trigger_rms=round(float(self.engine.speech_trigger_rms), 4),
            )
            self.after(0, lambda: self.calibrate_button.configure(text="🎯 Đo lại tiếng ồn phòng (1s)", state="normal"))
            self.after(0, lambda: self.noise_info.configure(
                text=f"Ồn phòng: {self.engine.ambient_noise_rms:.3f} · Bắt giọng: >{self.engine.speech_trigger_rms:.3f}"
            ))
            self.after(0, lambda: self.voice_detail.configure(text="🟢 Đã cân chỉnh tiếng ồn phòng! Đang lắng nghe...", text_color=SUCCESS))
            self.after(2500, self._reset_voice_status)
        except Exception as exc:
            self._log("noise_calibration_error", error=str(exc))
            self.after(0, lambda: self.calibrate_button.configure(text="🎯 Đo tiếng ồn phòng (1s)", state="normal"))
            self.after(0, lambda: self.voice_detail.configure(text=f"Lỗi đo ồn: {exc}", text_color=WARNING))
        finally:
            self.is_calibrating = False

    def _play_wake_chime(self):
        """Phát âm thanh thức tỉnh công nghệ cao khi trợ lý thức giấc."""
        play_sound(SOUND_GAMING_LOCK)

    def _listen_loop(self):
        vad_manager = SileroVADManager.get_instance()
        while True:
            time.sleep(0.09)
            if not self.listening or not self.engine.prototypes or self.is_calibrating:
                continue
            # Khi Antigravity đang bận học lệnh hoặc thực thi tác vụ, tạm dừng kích hoạt nhận diện
            if self.action_running:
                continue
            with self.buffer_lock:
                audio = self.buffer.copy()
            rms = calculate_rms(audio)
            self.after(0, lambda value=min(1.0, rms / max(self.engine.speech_trigger_rms * 2, 0.01)): self.voice_meter.set(value))
            now = time.time()

            # [XỬ LÝ TIẾNG NÓI - MÔ-ĐUN 1 & 2: VAD NOISE GATING & DUAL-TRIGGER]
            # Khi có tiếng quạt chạy bên cạnh, RMS luôn cao (>= 0.05-0.12).
            # Bắt buộc kiểm tra qua Silero VAD: Chỉ kích hoạt nhận diện khi THỰC SỰ CÓ TIẾNG NGƯỜI NÓI,
            # hoàn toàn không để tiếng quạt gió kích hoạt vòng nhận diện liên tục làm nghẽn hệ thống!
            has_voice, _, _ = vad_manager.get_speech_interval(audio, sr=16000, threshold=0.30)
            is_energy_active = (rms >= self.engine.speech_trigger_rms)

            should_trigger = has_voice and (is_energy_active or rms >= 0.025) and (now - self.last_detection > 0.8)
            if should_trigger:
                result = self.engine.classify_audio(audio)
                if result.get("status") == "OK":
                    self.last_detection = now
                    self._log(
                        "detection",
                        best_keyword=result.get("best_keyword"),
                        similarity=round(result.get("similarity", 0), 4),
                        required=round(result.get("required_similarity", 0), 4),
                        margin=result.get("winner_margin"),
                        required_margin=result.get("required_margin"),
                        accepted=result.get("is_match"),
                    )
                    self.after(0, lambda data=result: self._show_result(data))
                elif result.get("status") == "UNKNOWN":
                    self.last_detection = now
                    self._log(
                        "unknown_rejected",
                        best_keyword=result.get("best_keyword"),
                        similarity=round(result.get("similarity", 0), 4),
                        unk_sim=round(result.get("unk_sim", 0), 4),
                        unknown_margin=round(result.get("unknown_margin", 0), 4),
                    )
                    self.after(0, lambda data=result: self._show_result(data))
                elif result.get("status") == "NOISE_PROTOTYPE_MATCH":
                    # Không khóa cooldown nếu chỉ là tiếng ồn thoáng qua
                    self.last_detection = now - 0.85
                    self._log("noise_rejected", rms=round(rms, 4), noise_sim=round(result.get("noise_sim", 0), 4))
                    self.after(0, lambda: self.voice_detail.configure(text="Đã lọc tiếng ồn phòng", text_color=MUTED))

    def _show_result(self, result):
        now = time.time()
        is_awake = (not self.wake_slug) or (self.awake_until > now)

        # [XỬ LÝ TIẾNG NÓI - MÔ-ĐUN 6: CHẶN ĐỨNG HOÀN TOÀN TRẠNG THÁI UNKNOWN KHI SLEEPING]
        if result.get("status") == "UNKNOWN":
            if is_awake:
                remaining = int(math.ceil(max(0.0, self.awake_until - time.time()))) if self.awake_until > now else 0
                time_hint = f" (còn {remaining}s)" if remaining > 0 else ""
                self.voice_title.configure(text=f"Chưa rõ lệnh{time_hint}")
                self.voice_detail.configure(text="Chưa rõ từ khóa · Hãy nói lại câu lệnh", text_color=WARNING)
                self._set_orb_status(f"Chưa rõ{time_hint}", "Hãy nói lại câu lệnh")
                self.after(1400, self._reset_voice_status)
            else:
                # Đang ngủ: Giữ giao diện thanh thoát, không giật màn hình hoặc đổi tiêu đề cảnh báo
                self.voice_detail.configure(text="🔇 Âm thanh ngoài từ khóa · Đang lắng nghe wake word...", text_color=MUTED)
                self.after(1500, self._reset_voice_status)
            return

        slug = result.get("best_keyword", "")
        # Wake candidate BẮT BUỘC phải thỏa mãn status == OK, is_match == True (nằm trong bán kính R_k)
        wake_candidate = (
            bool(self.wake_slug)
            and slug == self.wake_slug
            and result.get("status") == "OK"
            and result.get("is_match", False)
            and result.get("similarity", 0) >= result.get("required_similarity", 0.935)
        )
        if not result.get("is_match") and not wake_candidate:
            if is_awake:
                similarity = result.get("similarity", 0) * 100
                required = result.get("required_similarity", self.engine.min_similarity_threshold) * 100
                remaining = int(math.ceil(max(0.0, self.awake_until - time.time()))) if self.awake_until > now else 0
                time_hint = f" (còn {remaining}s)" if remaining > 0 else ""
                self.voice_title.configure(text=f"Chưa rõ lệnh{time_hint}")
                self.voice_detail.configure(text=f"Độ khớp {similarity:.1f}% / cần {required:.1f}% · Hãy nói lại", text_color=WARNING)
                margin = result.get("winner_margin")
                required_margin = result.get("required_margin", self.engine.min_winner_margin)
                if similarity >= required and margin is not None and margin < required_margin:
                    self._set_orb_status(f"Chưa rõ{time_hint}", f"Lệnh quá giống nhau · {margin * 100:.1f}%")
                else:
                    self._set_orb_status(f"Chưa rõ{time_hint}", f"Khớp {similarity:.1f}% · Hãy nói lại")
                self.after(1400, self._reset_voice_status)
            else:
                # Đang ngủ: Giữ tĩnh lặng
                self.voice_detail.configure(text="Đã bỏ qua âm thanh không khớp wake word", text_color=MUTED)
                self.after(1500, self._reset_voice_status)
            return

        item = load_metadata().get(slug, {})
        if self.wake_slug:
            # [CHỐNG KÍCH HOẠT OAN DO NÓI XÀM / VIDEO YOUTUBE]:
            # Khi đã chọn Wake Word, hệ thống ở trạng thái Sleeping.
            # Bắt buộc người dùng phải gọi đúng Wake Word đạt chuẩn Bounded Prototype trước mới được kích hoạt các lệnh còn lại!
            if slug == self.wake_slug:
                self._log("wake_accepted", keyword=slug, similarity=round(result.get("similarity", 0), 4))
                # Hủy bỏ timer và ticker đếm ngược cũ nếu có (tránh hiện tượng đè timer gây lập tức sleeping)
                self._cancel_wake_timers()

                # Thiết lập thời hạn thức tỉnh 7.0 giây chuẩn hóa
                wake_duration = float(self.wake_window_seconds)
                self.awake_until = time.time() + wake_duration

                # Phát chuông thức tỉnh Siri tít tít
                self._play_wake_chime()

                if self.compact_mode:
                    self._show_wake_orb()

                self.voice_title.configure(text="I'm listening")
                self.connection.configure(text="●  Awake", text_color=SUCCESS)
                self.orb.configure(border_color=SUCCESS)
                self._tick_wake_countdown()

                # Lên lịch timeout duy nhất có quản lý Handle ID
                self.wake_timer_id = self.after(int(wake_duration * 1000), self._on_wake_timeout)
                return

            if time.time() > self.awake_until:
                wake_name = load_metadata().get(self.wake_slug, {}).get("name", self.wake_slug)
                self.voice_title.configure(text=f'Say “{wake_name}” first')
                self.voice_detail.configure(text=f"Ignored while sleeping · say “{wake_name}” to wake up", text_color=WARNING)
                self.after(2500, self._reset_voice_status)
                return

            # Nếu đang trong thời gian awake, nhận diện lệnh xong sẽ hủy timer và trở về trạng thái bình thường
            self._cancel_wake_timers()
            self.awake_until = 0.0

        self.voice_title.configure(text=item.get("name", slug))
        description = item.get("description", "").strip()
        self.voice_detail.configure(text=f"Recognized · {description}", text_color=SUCCESS)
        play_sound(SOUND_CLICK)
        self._set_orb_status(f"Đã nhận: {item.get('name', slug)}", "Đang chuẩn bị thực hiện")
        self._run_voice_action(slug, item)

    def _cancel_wake_timers(self):
        """Hủy toàn bộ timer timeout và đếm ngược wake window cũ để tránh race condition."""
        if self.wake_timer_id is not None:
            try:
                self.after_cancel(self.wake_timer_id)
            except Exception:
                pass
            self.wake_timer_id = None
        if self.wake_countdown_id is not None:
            try:
                self.after_cancel(self.wake_countdown_id)
            except Exception:
                pass
            self.wake_countdown_id = None

    def _tick_wake_countdown(self):
        """Cập nhật nhịp đếm lùi trực quan trên giao diện Full Window và Orb Overlay."""
        if self.awake_until <= 0.0:
            return
        remaining = int(math.ceil(max(0.0, self.awake_until - time.time())))
        if remaining > 0:
            self.voice_detail.configure(text=f"Awake ({remaining}s) · say any command", text_color=SUCCESS)
            if self.compact_mode:
                self._set_orb_status(f"Đang nghe ({remaining}s)", "Hãy nói một command")
            self.wake_countdown_id = self.after(500, self._tick_wake_countdown)

    def _on_wake_timeout(self):
        """Xử lý sự kiện hết hạn wake window đồng bộ độc lập cho Panel và Compact Mode."""
        self.wake_timer_id = None
        if self.action_running:
            return
        self.awake_until = 0.0
        self._cancel_wake_timers()
        if self.compact_mode:
            # Ở chế độ thu nhỏ: dọn dẹp tiến trình Orb overlay
            self._collapse()
        else:
            # Ở chế độ Bảng điều khiển chính (Panel): GIỮ NGUYÊN CỬA SỔ, chỉ trả trạng thái về Say "Tiểu Bảo"
            self._update_wake_status()

    def _run_voice_action(self, slug, item):
        """Run a recognized command and make every state visible in the UI."""
        description = item.get("description", "").strip()
        name = item.get("name", slug)
        if item.get("action_type") != "antigravity" or not description:
            self.voice_detail.configure(text="No Antigravity description configured", text_color=WARNING)
            self._set_orb_status("Thiếu hành động", "Command chưa có description")
            return
        if self.action_running:
            self.voice_detail.configure(text="Antigravity is already running another command", text_color=WARNING)
            self._set_orb_status("Đang bận", "Antigravity đang chạy lệnh khác")
            return
        now = time.time()
        if now - self.action_last_started.get(slug, 0.0) < 3.0:
            self.voice_detail.configure(text="Duplicate ignored · wait 3 seconds", text_color=WARNING)
            self._set_orb_status("Đã bỏ qua", "Command bị lặp trong 3 giây")
            return

        self.action_running = True
        self.action_last_started[slug] = now
        cached_profile = item.get("action_profile")
        if cached_profile:
            self.voice_title.configure(text=f"⚡ {name}")
            self.voice_detail.configure(text="⚡ Đang thực thi hành động đã học (tốc độ cao)…", text_color=ACCENT)
            self.connection.configure(text="●  Executing action", text_color=ACCENT)
            self.orb.configure(border_color=ACCENT)
            self._set_orb_status(f"Đang chạy: {name}", "Thực thi tác vụ đã học")
        else:
            # GIAO DIỆN HỌC LỆNH LẦN ĐẦU TIÊN CỦA ANTIGRAVITY
            self.voice_title.configure(text=f"🧠 Đang học: {name}")
            self.voice_detail.configure(
                text="⏳ Antigravity đang tự động tìm kiếm, lập trình và kiểm thử hành động lần đầu…",
                text_color=PURPLE
            )
            self.connection.configure(text="●  Learning action", text_color=PURPLE)
            self.orb.configure(border_color=PURPLE)
            self.voice_meter.configure(progress_color=PURPLE)
            self._set_orb_status("Đang học lệnh", f"Học tự động: {name}")

        def worker():
            try:
                if cached_profile:
                    execution = self.action_executor.execute(cached_profile)
                    if not execution.success:
                        raise RuntimeError(execution.error or "Cached action failed")
                    message = f"Done · cached method #{execution.selected_index + 1}"
                else:
                    resolver = AntigravityActionResolver(
                        PROJECT_ROOT,
                        allow_unattended_tools=True,
                    )
                    profile = resolver.resolve(description)
                    if profile.get("status") != "ready" or not profile.get("verified"):
                        detail = profile.get("error") or profile.get("message") or profile.get("status", "failed")
                        raise RuntimeError(f"Antigravity could not verify the action: {detail}")
                    metadata = load_metadata()
                    if slug in metadata:
                        metadata[slug]["action_profile"] = profile
                        with open(METADATA_FILE, "w", encoding="utf-8") as handle:
                            json.dump(metadata, handle, ensure_ascii=False, indent=2)
                    message = "Done · method learned for faster reuse"
                self.after(0, lambda: self._finish_voice_action(name, message, None))
            except Exception as exc:
                self.after(0, lambda error=str(exc): self._finish_voice_action(name, "", error))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_voice_action(self, name, message, error):
        self.action_running = False
        self.voice_title.configure(text=name)
        self.voice_meter.configure(progress_color=ACCENT)
        if error:
            self.voice_detail.configure(text=f"Failed · {error}", text_color=WARNING)
            self.connection.configure(text="●  Action failed", text_color=WARNING)
            self.orb.configure(border_color=WARNING)
            self._set_orb_status("Thực thi thất bại", error[:42])
        else:
            play_sound(SOUND_CONFIRM)
            self.voice_detail.configure(text=message, text_color=SUCCESS)
            self.connection.configure(text="●  Ready", text_color=SUCCESS)
            self.orb.configure(border_color=SUCCESS)
            self._set_orb_status("Đã hoàn tất", message[:42])
        self.after(4000, self._reset_voice_status)
        if self.compact_mode:
            self.after(4500, self._collapse)

    def _reset_voice_status(self):
        if self.action_running:
            return
        now = time.time()
        if self.wake_slug and self.awake_until > now:
            # Vẫn đang trong thời gian thức tỉnh! Không được đặt về Sleeping!
            self.voice_title.configure(text="I'm listening")
            self.connection.configure(text="●  Awake", text_color=SUCCESS)
            if hasattr(self, "orb"):
                self.orb.configure(border_color=SUCCESS)
            return
        self._update_wake_status()

    def _toggle_listening(self):
        self.listening = not self.listening
        self.voice_title.configure(text="Say a command" if self.listening else "Paused")
        self.connection.configure(text="●  Ready" if self.listening else "●  Paused", text_color=SUCCESS if self.listening else MUTED)

    def _bubble(self, role, message):
        user = role == "user"
        label = ctk.CTkLabel(self.messages, text=message, wraplength=620, justify="left", anchor="w", fg_color=TEXT if user else SURFACE, text_color=BG if user else TEXT, corner_radius=18, padx=14, pady=10, font=ctk.CTkFont(size=13))
        label.grid(sticky="e" if user else "w", padx=(100, 8) if user else (8, 100), pady=5)
        self.after(20, lambda: self.messages._parent_canvas.yview_moveto(1.0))
        return label

    def _send_chat(self):
        message = self.chat_input.get("1.0", "end").strip()
        if not message or self.chat_busy:
            return "break"
        self.chat_input.delete("1.0", "end")
        self._bubble("user", message)
        self.chat_reply = self._bubble("assistant", "Thinking…")
        self.chat_busy = True
        self.send_button.configure(state="disabled")
        if self.chat is None:
            self.chat = AntigravityChatSession(
                PROJECT_ROOT,
                allow_unattended_tools=True,
                on_event=lambda event: self.after(0, lambda e=event: self._chat_event(e)),
                on_error=lambda error: self.after(0, lambda err=error: self._chat_error(err)),
            )
            self.tool_switch.configure(state="disabled")
        try:
            self.chat.send(message)
            self.chat_status.configure(text="Connecting…")
        except Exception as exc:
            self._chat_error(str(exc))
        return "break"

    def _chat_event(self, event):
        kind = str(event.get("event") or event.get("type") or "").lower()
        if kind in {"init", "ready", "session_started"}:
            self.chat_status.configure(text="Connected")
            return

        text = ""
        step_update = event.get("step_update")
        if isinstance(step_update, dict):
            for key in ("text_delta", "delta", "text", "content"):
                val = step_update.get(key)
                if isinstance(val, str) and val:
                    text = val
                    break

        result = event.get("result")
        if isinstance(result, dict):
            resp = result.get("response")
            if isinstance(resp, str) and resp:
                if self.chat_reply:
                    current = self.chat_reply.cget("text")
                    if current == "Thinking…" or not current.strip():
                        self.chat_reply.configure(text=resp.strip())

        if not text:
            for key in ("text_delta", "delta", "text", "content", "message"):
                val = event.get(key)
                if isinstance(val, str) and val:
                    text = val
                    break

        if text and self.chat_reply:
            current = self.chat_reply.cget("text")
            new_text = ("" if current == "Thinking…" else current) + text
            self.chat_reply.configure(text=new_text)

        if kind in {"result", "done", "turn_completed", "completed"}:
            self.chat_busy = False
            self.send_button.configure(state="normal")
            self.chat_status.configure(text="Connected")

    def _chat_error(self, error):
        self.chat_busy = False
        self.send_button.configure(state="normal")
        self.chat_status.configure(text="Offline")
        if self.chat_reply:
            self.chat_reply.configure(text=f"Connection error: {error}")

    def _close(self):
        self.listening = False
        if self.orb_process is not None:
            try:
                if self.orb_process.poll() is None:
                    self.orb_process.terminate()
            except Exception:
                pass
            self.orb_process = None
        for signal in (OPEN_UI_SIGNAL, SHUTDOWN_SIGNAL):
            try:
                if os.path.exists(signal):
                    os.remove(signal)
            except OSError:
                pass
        if self.chat:
            try:
                self.chat.close()
            except Exception:
                pass
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        try:
            self.destroy()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    VoiceAppV2().mainloop()
