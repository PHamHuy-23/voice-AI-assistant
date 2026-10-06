"""Antigravity action compiler and cached Windows action executor."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


DEMO_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = DEMO_DIR / "prompts" / "antigravity_action_resolver.md"
ACTION_SCHEMA_PATH = DEMO_DIR / "prompts" / "action_profile.schema.json"


@dataclass(frozen=True)
class ActionExecutionResult:
    success: bool
    selected_index: int
    error: str = ""


class AntigravityActionResolver:
    """Compile a natural-language description into a verified action profile."""

    def __init__(
        self,
        workspace: str,
        model: str = "gemini-3.8-flash-low",
        effort: str = "low",
        allow_unattended_tools: bool = False,
    ):
        self.workspace = str(Path(workspace).resolve())
        self.model = model
        self.effort = effort
        self.allow_unattended_tools = allow_unattended_tools

    @staticmethod
    def find_executable() -> Optional[str]:
        discovered = shutil.which("agy") or shutil.which("agy.exe")
        fallback = Path.home() / "AppData" / "Local" / "agy" / "bin" / "agy.exe"
        return discovered or (str(fallback) if fallback.exists() else None)

    def build_prompt(self, description: str) -> str:
        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        return f"{system_prompt}\n\nUSER_DESCRIPTION\n{description.strip()}"

    def build_command(self, description: str) -> list[str]:
        executable = self.find_executable()
        if not executable:
            raise FileNotFoundError("Antigravity CLI (agy) không có trong PATH.")
        args = [
            executable,
            "--model", self.model,
            "--effort", self.effort,
            "--disable-slash-commands",
            "--output-format", "json",
            "--json-schema", str(ACTION_SCHEMA_PATH),
            "--print", self.build_prompt(description),
            "--print-timeout", "5m",
        ]
        if self.allow_unattended_tools:
            args.insert(1, "--dangerously-skip-permissions")
        return args

    def resolve(self, description: str, timeout_seconds: int = 330) -> dict[str, Any]:
        if not description.strip():
            raise ValueError("Description không được để trống.")
        completed = subprocess.run(
            self.build_command(description),
            cwd=self.workspace,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"Antigravity exit code {completed.returncode}")
        payload = json.loads(completed.stdout)
        # Depending on CLI version, structured final output can be wrapped.
        for key in ("structured_output", "result", "response", "output"):
            if isinstance(payload, dict) and isinstance(payload.get(key), dict):
                payload = payload[key]
                break
        return payload

    def stream_chat(self, messages: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Foundation for the future chat window using Antigravity stream-json."""
        executable = self.find_executable()
        if not executable:
            raise FileNotFoundError("Antigravity CLI (agy) không có trong PATH.")
        args = [
            executable,
            "--model", self.model,
            "--effort", self.effort,
            "--disable-slash-commands",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
        ]
        process = subprocess.Popen(
            args,
            cwd=self.workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        assert process.stdin is not None and process.stdout is not None
        for message in messages:
            process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if line:
                yield json.loads(line)


class AntigravityChatSession:
    """Persistent NDJSON Antigravity process for a desktop chat UI."""

    def __init__(
        self,
        workspace: str,
        model: str = "gemini-3.8-flash-low",
        effort: str = "low",
        allow_unattended_tools: bool = False,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.workspace = str(Path(workspace).resolve())
        self.model = model
        self.effort = effort
        self.allow_unattended_tools = allow_unattended_tools
        self.on_event = on_event
        self.on_error = on_error
        self.process: Optional[subprocess.Popen[str]] = None
        self._write_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        if self.is_running:
            return
        executable = AntigravityActionResolver.find_executable()
        if not executable:
            raise FileNotFoundError("Antigravity CLI (agy) không có trong PATH.")
        args = [
            executable,
            "--dangerously-skip-permissions",
            "--model", self.model,
            "--effort", self.effort,
            "--disable-slash-commands",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
        ]
        self.process = subprocess.Popen(
            args,
            cwd=self.workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._reader_thread = threading.Thread(target=self._read_events, daemon=True, name="agy-chat-reader")
        self._reader_thread.start()
        threading.Thread(target=self._read_errors, daemon=True, name="agy-chat-errors").start()

    def send(self, message: str) -> None:
        message = message.strip()
        if not message:
            return
        if not self.is_running:
            self.start()
        assert self.process is not None and self.process.stdin is not None
        event = {"event": "user", "message": {"content": message}}
        with self._write_lock:
            self.process.stdin.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.process.stdin.flush()

    def _read_events(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"event": "raw", "text": line.rstrip()}
            if self.on_event:
                self.on_event(event)

    def _read_errors(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            text = line.strip()
            if text and self.on_error:
                self.on_error(text)

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()


class CachedActionExecutor:
    """Execute primary then fallbacks without invoking an AI model."""

    ALLOWED_TYPES = {"open_url", "open_app", "open_path", "open_settings"}

    def execute(
        self,
        profile: dict[str, Any],
        on_attempt: Optional[Callable[[int, dict[str, Any]], None]] = None,
    ) -> ActionExecutionResult:
        if profile.get("status") != "ready" or not profile.get("verified"):
            return ActionExecutionResult(False, -1, "Action profile chưa được xác minh.")
        actions = [profile.get("primary", {})] + list(profile.get("fallbacks", []))
        errors: list[str] = []
        for index, action in enumerate(actions):
            if action.get("type") not in self.ALLOWED_TYPES:
                errors.append(f"action {index}: type không được cache")
                continue
            executable = action.get("executable", "").strip()
            if not executable:
                errors.append(f"action {index}: thiếu executable")
                continue
            if on_attempt:
                on_attempt(index, action)
            try:
                subprocess.Popen(
                    [executable, *action.get("arguments", [])],
                    cwd=action.get("working_directory") or None,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                return ActionExecutionResult(True, index)
            except (OSError, ValueError) as exc:
                errors.append(f"action {index}: {exc}")
        return ActionExecutionResult(False, -1, "; ".join(errors))
