"""Safe, non-blocking integration layer for Codex CLI voice actions."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class CodexRunResult:
    command_slug: str
    success: bool
    exit_code: int
    output: str
    error: str
    log_path: str


class CodexRunner:
    """Runs one Codex task at a time without blocking the Tk event loop."""

    def __init__(self, workspace: str, log_dir: str, timeout_seconds: int = 900):
        self.workspace = str(Path(workspace).resolve())
        self.log_dir = Path(log_dir)
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._active_slug: Optional[str] = None

    @property
    def active_slug(self) -> Optional[str]:
        with self._lock:
            return self._active_slug

    @staticmethod
    def find_executable() -> Optional[str]:
        # npm exposes a .cmd shim on Windows; subprocess can launch it directly.
        names = ("codex.cmd", "codex.exe", "codex") if os.name == "nt" else ("codex",)
        return next((path for name in names if (path := shutil.which(name))), None)

    def start(
        self,
        command_slug: str,
        prompt: str,
        on_started: Optional[Callable[[str], None]] = None,
        on_finished: Optional[Callable[[CodexRunResult], None]] = None,
    ) -> bool:
        prompt = prompt.strip()
        if not prompt:
            return False

        with self._lock:
            if self._active_slug is not None:
                return False
            self._active_slug = command_slug

        if on_started:
            on_started(command_slug)

        worker = threading.Thread(
            target=self._run,
            args=(command_slug, prompt, on_finished),
            daemon=True,
            name=f"codex-{command_slug}",
        )
        worker.start()
        return True

    def _run(
        self,
        command_slug: str,
        prompt: str,
        on_finished: Optional[Callable[[CodexRunResult], None]],
    ) -> None:
        executable = self.find_executable()
        exit_code = 127
        stdout = ""
        stderr = "Codex CLI chưa được cài đặt hoặc không có trong PATH."

        if executable:
            args = [
                executable,
                "exec",
                "--approve-for-me",
                "--color",
                "never",
                "--cd",
                self.workspace,
                prompt,
            ]
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                completed = subprocess.run(
                    args,
                    cwd=self.workspace,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    creationflags=creationflags,
                    check=False,
                )
                exit_code = completed.returncode
                stdout = completed.stdout.strip()
                stderr = completed.stderr.strip()
            except subprocess.TimeoutExpired as exc:
                exit_code = 124
                stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
                stderr = f"Codex đã bị dừng sau {self.timeout_seconds} giây do quá thời gian."
            except Exception as exc:  # Surface launch/auth errors to the UI and log.
                exit_code = 1
                stderr = f"Không thể chạy Codex CLI: {exc}"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"{stamp}_{command_slug}.log"
        log_path.write_text(
            f"COMMAND: {command_slug}\nPROMPT:\n{prompt}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n",
            encoding="utf-8",
        )

        result = CodexRunResult(
            command_slug=command_slug,
            success=exit_code == 0,
            exit_code=exit_code,
            output=stdout,
            error=stderr,
            log_path=str(log_path),
        )
        with self._lock:
            self._active_slug = None
        if on_finished:
            on_finished(result)
