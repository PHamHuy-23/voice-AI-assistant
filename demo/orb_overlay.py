"""Independent animated wake overlay with local status and shutdown signals."""

from __future__ import annotations

import json
import math
import sys
import time
import tkinter as tk
from pathlib import Path


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
        shutdown_signal.parent.mkdir(parents=True, exist_ok=True)
        shutdown_signal.write_text(str(time.time()), encoding="utf-8")
        root.destroy()

    for item in (core, outer_ring, *bars):
        canvas.tag_bind(item, "<Button-1>", open_ui)
    for item in (close_bg, close_text):
        canvas.tag_bind(item, "<Button-1>", shutdown)

    started = time.monotonic()
    colors = ("#b9ddff", "#d8c7ff", "#ffffff", "#c7e7ff")
    state = {"frame": 0, "status_mtime": 0.0, "has_result": False}

    def show_status(message):
        canvas.itemconfigure(status_text, text=message[:34])
        for item in status_items:
            canvas.itemconfigure(item, state="normal")

    def hide_status():
        for item in status_items:
            canvas.itemconfigure(item, state="hidden")

    def animate():
        if time.monotonic() - started >= timeout:
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
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                status_title = str(payload.get("title", ""))
                if status_title == "Không nhận diện":
                    state["has_result"] = True
                    show_status(str(payload.get("detail", "Không nhận diện")))
                elif status_title not in {"", "Đang nghe"}:
                    state["has_result"] = True
                    hide_status()
                state["status_mtime"] = mtime
        except (OSError, json.JSONDecodeError):
            pass
        if root.winfo_exists():
            root.after(120, refresh_status)

    def delayed_prompt():
        if not state["has_result"]:
            show_status("Hãy nói command")

    root.after(5000, delayed_prompt)
    animate()
    refresh_status()
    root.mainloop()


if __name__ == "__main__":
    main()
