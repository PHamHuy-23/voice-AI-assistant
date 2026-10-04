"""Compile one voice-command description into a machine-specific action cache."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from demo.antigravity_runner import AntigravityActionResolver


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.replace("đ", "d").replace("Đ", "D"))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return "_".join(part for part in "".join(c if c.isalnum() else " " for c in ascii_value).split())[:80]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Resolve and cache a Windows voice action with Antigravity.")
    parser.add_argument("description", help="Natural-language action description.")
    parser.add_argument("--name", help="Cache name; defaults to a slug of the description.")
    parser.add_argument("--model", default="gemini-3.8-flash-low")
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument(
        "--allow-unattended-tools",
        action="store_true",
        help="Pass --dangerously-skip-permissions to Antigravity for this resolution run.",
    )
    args = parser.parse_args()

    resolver = AntigravityActionResolver(
        str(PROJECT_ROOT),
        model=args.model,
        effort=args.effort,
        allow_unattended_tools=args.allow_unattended_tools,
    )
    profile = resolver.resolve(args.description)
    profile["description"] = args.description
    profile["resolved_at"] = datetime.now(timezone.utc).isoformat()

    cache_dir = PROJECT_ROOT / "demo" / "action_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = slugify(args.name or args.description) or "voice_action"
    output_path = cache_dir / f"{cache_name}.json"
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    print(output_path)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0 if profile.get("status") == "ready" and profile.get("verified") else 2


if __name__ == "__main__":
    raise SystemExit(main())
