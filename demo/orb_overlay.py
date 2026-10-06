"""Independent animated wake overlay with local status and shutdown signals."""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

try:
    import winsound
except ImportError:
    winsound = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOUND_DIR = PROJECT_ROOT / "demo" / "assets" / "sounds"
SOUND_CLICK = str(SOUND_DIR / "click.wav")
SOUND_CONFIRM = str(SOUND_DIR / "confirm.wav")
SOUND_GAMING_LOCK = str(SOUND_DIR / "gaming_lock.wav")


def play_sound(path: str) -> None:
    """Phát âm thanh UI hiệu ứng công nghệ cao trên luồng nền daemon."""
    if winsound is None or not Path(path).exists():
        return

    def _worker():
        try:
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def play_wake_chime() -> None:
    """Phát âm thanh thức tỉnh công nghệ cao khi trợ lý thức giấc."""
    play_sound(SOUND_GAMING_LOCK)


def main() -> None:
    open_signal = Path(sys.argv[1]).resolve()
    status_path = Path(sys.argv[2]).resolve()
    shutdown_signal = Path(sys.argv[3]).resolve()
    timeout = float(sys.argv[4]) if len(sys.argv) > 4 else 15.0
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    transparent = "#010101"
    root.configure(bg=transparent)
    try:
        root.attributes("-transparentcolor", transparent)
        root.attributes("-toolwindow", True)
    except tk.TclError:
        pass

    width, height = 300, 88
    x = root.winfo_screenwidth() - width - 28
    y = root.winfo_screenheight() - height - 66
    root.geometry(f"{width}x{height}+{x}+{y}")
    canvas = tk.Canvas(root, width=width, height=height, bg=transparent, highlightthickness=0)
    canvas.pack()
    status_left = canvas.create_oval(10, 26, 48, 64, fill="#ffffff", outline="#e5e5ea", width=1, state="hidden")
    status_right = canvas.create_oval(190, 26, 228, 64, fill="#ffffff", outline="#e5e5ea", width=1, state="hidden")
    status_middle = canvas.create_rectangle(29, 26, 209, 64, fill="#ffffff", outline="#ffffff", state="hidden")
    status_text = canvas.create_text(27, 45, text="Hãy nói command", fill="#1d1d1f", anchor="w", font=("Segoe UI", 9, "bold"), state="hidden")
    status_items = (status_left, status_right, status_middle, status_text)

    center_x, center_y, radius = 264, 45, 30
    outer_ring = canvas.create_oval(center_x-radius, center_y-radius, center_x+radius, center_y+radius, fill="", outline="#c7e7ff", width=3)
    core = canvas.create_oval(center_x-24, center_y-24, center_x+24, center_y+24, fill="#ffffff", outline="#ffffff")
    bars = [canvas.create_line(center_x + i * 7, center_y-6, center_x + i * 7, center_y+6, fill="#111114", width=3, capstyle="round") for i in (-1, 0, 1)]
    close_bg = canvas.create_oval(278, 4, 298, 24, fill="#ffffff", outline="#e5e5ea")
    close_text = canvas.create_text(288, 14, text="×", fill="#6e6e73", font=("Segoe UI", 10, "bold"))

    def open_ui(_event=None):
        open_signal.parent.mkdir(parents=True, exist_ok=True)
        open_signal.write_text(str(time.time()), encoding="utf-8")
        root.destroy()

    def shutdown(_event=None):
        # Nút [x] trên Orb overlay chỉ đóng cửa sổ nổi, không làm sập ứng dụng chính
        root.destroy()

    for item in (core, outer_ring, *bars):
        canvas.tag_bind(item, "<Button-1>", open_ui)
    for item in (close_bg, close_text):
        canvas.tag_bind(item, "<Button-1>", shutdown)

    started = time.monotonic()
    colors = ("#b9ddff", "#d8c7ff", "#ffffff", "#c7e7ff")
    state = {"frame": 0, "status_mtime": 0.0, "has_result": False, "last_activity": time.monotonic()}

    def show_status(message):
        canvas.itemconfigure(status_text, text=message[:34])
        for item in status_items:
            canvas.itemconfigure(item, state="normal")

    def hide_status():
        for item in status_items:
            canvas.itemconfigure(item, state="hidden")

    def animate():
        # Chỉ tự đóng nếu sau `timeout` giây mà không có bất kỳ hoạt động nào từ app
        if time.monotonic() - state["last_activity"] >= timeout:
            root.destroy()
            return
        frame = state["frame"]
        phase = frame * 0.16
        pulse = (math.sin(phase) + 1.0) / 2.0
        outer_radius = radius - pulse * 5
        core_radius = 24 - (1.0 - pulse) * 2
        canvas.coords(outer_ring, center_x-outer_radius, center_y-outer_radius, center_x+outer_radius, center_y+outer_radius)
        canvas.coords(core, center_x-core_radius, center_y-core_radius, center_x+core_radius, center_y+core_radius)
        canvas.itemconfigure(outer_ring, outline=colors[(frame // 7) % len(colors)], width=2 + int(pulse * 2))
        for index, bar in enumerate(bars):
            bar_height = 7 + 9 * ((math.sin(phase * 1.7 + index * 1.4) + 1.0) / 2.0)
            bar_x = center_x + (index - 1) * 8
            canvas.coords(bar, bar_x, center_y - bar_height / 2, bar_x, center_y + bar_height / 2)
        state["frame"] += 1
        root.after(55, animate)

    def refresh_status():
        try:
            mtime = status_path.stat().st_mtime
            if mtime != state["status_mtime"]:
                state["last_activity"] = time.monotonic()
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                status_title = str(payload.get("title", ""))
                status_detail = str(payload.get("detail", ""))

                if status_title.startswith("Đã nhận:"):
                    state["has_result"] = True
                    cmd_name = status_title.replace("Đã nhận:", "").strip()
                    show_status(f"✨ {cmd_name}")
                elif status_title == "Không nhận diện":
                    state["has_result"] = True
                    show_status(f"❓ {status_detail[:24]}")
                elif status_title == "Đang học lệnh" or "học" in status_title.lower():
                    state["has_result"] = True
                    show_status(f"🧠 {status_detail[:24]}")
                elif status_title == "Đang thực thi" or status_title == "Đang chuẩn bị thực hiện":
                    state["has_result"] = True
                    show_status(f"⚡ {status_detail[:24]}")
                elif status_title == "Đã hoàn tất":
                    state["has_result"] = True
                    show_status("✅ Hoàn thành!")
                elif status_title == "Thực thi thất bại":
                    state["has_result"] = True
                    show_status(f"❌ {status_detail[:24]}")
                elif status_title.startswith("Đang nghe"):
                    show_status(f"🎙️ {status_title}...")
                elif status_title:
                    state["has_result"] = True
                    show_status(status_title[:28])

                state["status_mtime"] = mtime
        except (OSError, json.JSONDecodeError):
            pass
        if root.winfo_exists():
            root.after(100, refresh_status)

    def delayed_prompt():
        if not state["has_result"]:
            show_status("🎙️ Hãy nói command")

    play_wake_chime()
    root.after(100, delayed_prompt)
    animate()
    refresh_status()
    root.mainloop()


if __name__ == "__main__":
    main()
