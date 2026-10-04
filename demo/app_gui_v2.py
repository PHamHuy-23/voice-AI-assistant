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
SETTINGS_FILE = os.path.join(KEYWORDS_DIR, "settings.json")
RUNTIME_LOG = os.path.join(PROJECT_ROOT, "demo", "logs", "voice_runtime.log")
OPEN_UI_SIGNAL = os.path.join(PROJECT_ROOT, "demo", "logs", "open_ui.signal")
ORB_STATUS_FILE = os.path.join(PROJECT_ROOT, "demo", "logs", "orb_status.json")
SHUTDOWN_SIGNAL = os.path.join(PROJECT_ROOT, "demo", "logs", "shutdown.signal")


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


class VoiceAppV2(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Voice")
        self.geometry("820x600")
        self.minsize(740, 540)
        self.configure(fg_color=BG)
        self.protocol("WM_DELETE_WINDOW", self._collapse)

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
        self.wake_window_seconds = 15
        self.chat = None
        self.chat_busy = False
        self.chat_reply = None
        self.compact_mode = False
        self.orb_process = None

        self._build_shell()
        self._show_page("voice")
        self._refresh_commands()
        self._start_audio()
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

        self.orb_window = ctk.CTkToplevel(self)
        self.orb_window.withdraw()
        self.orb_window.overrideredirect(True)
        self.orb_window.attributes("-topmost", True)
        self.orb_window.configure(fg_color="#010101")
        try:
            self.orb_window.attributes("-transparentcolor", "#010101")
        except Exception:
            pass
        self.compact = ctk.CTkFrame(self.orb_window, width=68, height=68, fg_color="#ffffff", corner_radius=34, border_width=3, border_color="#d9d9de")
        self.compact.pack_propagate(False)
        self.compact_orb = ctk.CTkButton(self.compact, text="", width=56, height=56, corner_radius=28, fg_color="#ffffff", hover_color="#f2f2f7", command=self._expand)
        self.compact_orb.place(relx=0.5, rely=0.5, anchor="center")
        self._glow_colors = ("#ffffff", "#d9d9de", "#b9ddff", "#ffffff")
        self._glow_index = 0
        self.after(180, self._pulse_compact)

    def _pulse_compact(self):
        if not self.winfo_exists():
            return
        if self.orb_window.state() != "withdrawn" and self.listening:
            color = self._glow_colors[self._glow_index % len(self._glow_colors)]
            self._glow_index += 1
            self.compact.configure(border_color=color)
        self.after(420, self._pulse_compact)

    def _collapse(self):
        self.compact_mode = True
        self.orb_window.withdraw()
        if self.orb_process is not None and self.orb_process.poll() is None:
            self.orb_process.terminate()
        self.orb_process = None
        self.rail.grid_remove()
        self.content.grid_remove()
        self.overrideredirect(True)
        self.withdraw()

    def _show_wake_orb(self):
        self.compact_mode = True
        if self.orb_process is not None and self.orb_process.poll() is None:
            return
        overlay = os.path.join(PROJECT_ROOT, "demo", "orb_overlay.py")
        try:
            for signal in (OPEN_UI_SIGNAL, SHUTDOWN_SIGNAL):
                if os.path.exists(signal):
                    os.remove(signal)
            self._set_orb_status("Đang nghe", "Hãy nói một command")
            self.orb_process = subprocess.Popen(
                [sys.executable, overlay, OPEN_UI_SIGNAL, ORB_STATUS_FILE, SHUTDOWN_SIGNAL, "120"],
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
        self.orb_window.withdraw()
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
            text = ctk.CTkFrame(row, fg_color="transparent")
            text.pack(side="left", fill="x", expand=True, padx=18, pady=14)
            ctk.CTkLabel(text, text=item.get("name", slug), font=ctk.CTkFont(size=15, weight="bold"), text_color=TEXT).pack(anchor="w")
            ctk.CTkLabel(text, text=item.get("description", ""), font=ctk.CTkFont(size=12), text_color=MUTED, wraplength=580, justify="left").pack(anchor="w", pady=(3, 0))
            ctk.CTkButton(row, text="Remove", width=72, fg_color="transparent", hover_color=SURFACE_2, text_color=MUTED, command=lambda key=slug: self._remove_command(key)).pack(side="right", padx=14)

    def _select_wake_word(self, label):
        self.wake_slug = self._wake_choices.get(label, "")
        self.awake_until = 0.0
        save_settings({"wake_word": self.wake_slug})
        self._update_wake_status()

    def _update_wake_status(self):
        if not hasattr(self, "voice_title"):
            return
        if self.wake_slug:
            item = load_metadata().get(self.wake_slug, {})
            name = item.get("name", self.wake_slug)
            self.voice_title.configure(text=f'Say “{name}”')
            self.voice_detail.configure(text="Sleeping · waiting for wake word", text_color=MUTED)
            self.connection.configure(text="●  Sleeping", text_color=MUTED)
        else:
            self.voice_title.configure(text="Say a command")
            self.voice_detail.configure(text="Wake word is off", text_color=MUTED)
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

    def _listen_loop(self):
        while True:
            time.sleep(0.09)
            if not self.listening or not self.engine.prototypes:
                continue
            with self.buffer_lock:
                audio = self.buffer.copy()
            rms = calculate_rms(audio)
            self.after(0, lambda value=min(1.0, rms / max(self.engine.speech_trigger_rms * 2, 0.01)): self.voice_meter.set(value))
            now = time.time()
            if rms >= self.engine.speech_trigger_rms and now - self.last_detection > 1.6:
                self.last_detection = now
                result = self.engine.classify_audio(audio)
                if result.get("status") == "OK":
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

    def _show_result(self, result):
        slug = result.get("best_keyword", "")
        wake_candidate = (
            bool(self.wake_slug)
            and slug == self.wake_slug
            and result.get("similarity", 0) >= result.get("required_similarity", self.engine.min_similarity_threshold)
        )
        if not result.get("is_match") and not wake_candidate:
            self.voice_title.configure(text="Not recognized")
            similarity = result.get("similarity", 0) * 100
            required = result.get("required_similarity", self.engine.min_similarity_threshold) * 100
            self.voice_detail.configure(text=f"Rejected · similarity {similarity:.1f}% / required {required:.1f}%", text_color=WARNING)
            if self.awake_until > time.time():
                margin = result.get("winner_margin")
                required_margin = result.get("required_margin", self.engine.min_winner_margin)
                if similarity >= required and margin is not None and margin < required_margin:
                    self._set_orb_status("Không nhận diện", f"Lệnh quá giống nhau · {margin * 100:.1f}% / {required_margin * 100:.0f}%")
                else:
                    self._set_orb_status("Không nhận diện", f"Độ giống {similarity:.1f}% · cần {required:.1f}%")
            return
        item = load_metadata().get(slug, {})
        if self.wake_slug:
            if slug == self.wake_slug:
                self._log("wake_accepted", keyword=slug, similarity=round(result.get("similarity", 0), 4))
                self.awake_until = time.time() + self.wake_window_seconds
                self._show_wake_orb()
                self.voice_title.configure(text="I'm listening")
                self.voice_detail.configure(text=f"Awake for {self.wake_window_seconds} seconds · say a command", text_color=SUCCESS)
                self.connection.configure(text="●  Awake", text_color=SUCCESS)
                wake_deadline = self.awake_until
                self.after(self.wake_window_seconds * 1000, lambda deadline=wake_deadline: self._hide_if_wake_expired(deadline))
                return
            if time.time() > self.awake_until:
                wake_name = load_metadata().get(self.wake_slug, {}).get("name", self.wake_slug)
                self.voice_title.configure(text=f'Say “{wake_name}” first')
                self.voice_detail.configure(text="Command ignored while sleeping", text_color=WARNING)
                return
            self.awake_until = 0.0
        self.voice_title.configure(text=item.get("name", slug))
        description = item.get("description", "").strip()
        self.voice_detail.configure(text=f"Recognized · {description}", text_color=SUCCESS)
        self._set_orb_status(f"Đã nhận: {item.get('name', slug)}", "Đang chuẩn bị thực hiện")
        self._run_voice_action(slug, item)

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
        if now - self.action_last_started.get(slug, 0.0) < 10:
            self.voice_detail.configure(text="Duplicate ignored · wait 10 seconds", text_color=WARNING)
            self._set_orb_status("Đã bỏ qua", "Command bị lặp trong 10 giây")
            return

        self.action_running = True
        self.action_last_started[slug] = now
        cached_profile = item.get("action_profile")
        if cached_profile:
            self.voice_detail.configure(text="Running cached action…", text_color=ACCENT)
        else:
            self.voice_detail.configure(text="Antigravity is learning and executing this command…", text_color=ACCENT)
        self.connection.configure(text="●  Antigravity running", text_color=ACCENT)
        self._set_orb_status(f"Đang chạy: {name}", "Antigravity đang thực hiện")

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
        if error:
            self.voice_detail.configure(text=f"Failed · {error}", text_color=WARNING)
            self.connection.configure(text="●  Action failed", text_color=WARNING)
            self._set_orb_status("Thực thi thất bại", error[:42])
        else:
            self.voice_detail.configure(text=message, text_color=SUCCESS)
            self.connection.configure(text="●  Ready", text_color=SUCCESS)
            self._set_orb_status("Đã hoàn tất", message[:42])
        self.after(6000, self._reset_voice_status)
        self.after(6500, self._collapse)

    def _hide_if_wake_expired(self, deadline):
        if self.action_running or not self.compact_mode:
            return
        if self.awake_until == deadline and time.time() >= deadline:
            self.awake_until = 0.0
            self._collapse()

    def _reset_voice_status(self):
        if self.action_running:
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
            self.chat = AntigravityChatSession(PROJECT_ROOT, allow_unattended_tools=bool(self.tool_switch.get()), on_event=lambda event: self.after(0, lambda: self._chat_event(event)), on_error=lambda error: self.after(0, lambda: self._chat_error(error)))
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
        text = ""
        for key in ("text_delta", "delta", "text", "content", "message"):
            if isinstance(event.get(key), str):
                text = event[key]
                break
        if text and self.chat_reply:
            current = self.chat_reply.cget("text")
            self.chat_reply.configure(text=("" if current == "Thinking…" else current) + text)
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
        if self.chat:
            self.chat.close()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    VoiceAppV2().mainloop()
