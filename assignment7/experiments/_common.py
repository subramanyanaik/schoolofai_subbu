"""Shared plumbing for the experiment scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CHECKPOINTS = ROOT / "checkpoints"

sys.path.insert(0, str(ROOT / "src"))

from kronecker_v2.data import force_utf8_stdout  # noqa: E402

force_utf8_stdout()


def write_json(name: str, payload) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n-> wrote {path.relative_to(ROOT)}")
    return path


def read_json(name: str):
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def merge_json(name: str, updates: dict, key: str = "runs") -> dict:
    """Trap 8: sweeps must merge, never overwrite.

    A timed-out sweep once rewrote its own results file with a single partial
    run, and the next invocation merged that partial into its output -- five
    completed runs gone.
    """
    existing = read_json(name) or {}
    runs = dict(existing.get(key, {}))
    runs.update(updates)
    existing[key] = runs
    write_json(name, existing)
    return existing


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def table(rows, headers) -> None:
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
