"""
runlog.py — the execution log.

`run.log` has to carry the whole sequence of events AND the pass/fail markers
the assignment names. Two things make it useful rather than decorative:

  * It is append-only across subprocesses. The crash phase, the resume phase
    and the fork phase are separate OS processes; they all append to the same
    file, so the log reads as one narrative including the process that died.

  * `[PASS]` and `[FAIL]` lines carry the numbers behind them. A line saying
    `[PASS] resume_next_batch_matched` proves nothing on its own; the same line
    with `expected_next_step=12 actual_next_step=12 hashes_compared=4
    mismatches=0` can be checked against the ledger by a reader who does not
    trust it.

Every check that emits a marker also returns its result to the caller, so the
evidence bundle is built from the same values rather than by parsing this file
back. The log is for humans; `evidence.json` is for machines.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

PASS = "PASS"
FAIL = "FAIL"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{v}" for k, v in sorted(value.items())) + "}"
    return str(value)


class RunLog:
    def __init__(self, path: Path | str, phase: str = "-", echo: bool = True,
                 append: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.phase = phase
        self.echo = echo
        self.markers: list = []
        if not append and self.path.exists():
            self.path.unlink()

    # -- primitives --------------------------------------------------------
    def _write(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
            fh.flush()
        if self.echo:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def event(self, event: str, **fields) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        parts = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
        self._write(f"{ts}  EVENT  phase={self.phase}  {event}"
                    + (f"  {parts}" if parts else ""))

    def info(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._write(f"{ts}  INFO   phase={self.phase}  {message}")

    def section(self, title: str) -> None:
        self._write("")
        self._write(f"=== {title} ".ljust(78, "="))

    def check(self, name: str, ok: bool, **fields) -> bool:
        """Emit a [PASS]/[FAIL] marker with the numbers that justify it."""
        tag = PASS if ok else FAIL
        parts = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
        self._write(f"[{tag}] {name}" + (f"  {parts}" if parts else ""))
        self.markers.append(dict(name=name, result=tag, phase=self.phase,
                                 detail={k: v for k, v in fields.items()}))
        return ok

    # -- convenience used as a callback by the trainer --------------------
    def __call__(self, event: str, **fields) -> None:
        self.event(event, **fields)

    def sub(self, phase: str) -> "RunLog":
        child = RunLog(self.path, phase=phase, echo=self.echo, append=True)
        child.markers = self.markers
        return child


def read_markers(path: Path | str) -> list:
    """Parse [PASS]/[FAIL] lines back out of a run.log.

    Used by the evidence generator, which is required to work from the written
    artefacts rather than from anything still in memory.
    """
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not (line.startswith("[PASS]") or line.startswith("[FAIL]")):
                continue
            tag = PASS if line.startswith("[PASS]") else FAIL
            rest = line[len(f"[{tag}]"):].strip()
            bits = rest.split()
            name = bits[0] if bits else ""
            detail = {}
            for b in bits[1:]:
                if "=" in b:
                    k, v = b.split("=", 1)
                    detail[k] = v
            out.append(dict(name=name, result=tag, detail=detail))
    return out
