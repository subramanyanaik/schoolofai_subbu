"""
ledger.py — the append-only record of what actually happened.

"Even when the planned order is generated from a seed and an index, the run
still needs an append-only record of the actual consumed stream."

Two design choices carry most of the weight here.

1. THE LOG IS HASH-CHAINED. Every event stores the chain hash of its
   predecessor, so `event_hash = H(prev_hash || event)`. Editing, reordering or
   deleting any historical event invalidates every hash after it, and
   `verify_chain` finds the exact index where the break occurs. Without this,
   "append-only" is a naming convention enforced by nothing.

2. ROLLBACK APPENDS, IT DOES NOT DELETE. When a crashed run resumes from a
   checkpoint, the batches consumed after that checkpoint never reached the
   weights, so they must be re-consumed. The tempting implementation is to
   truncate the log back to the checkpoint's offset. That is wrong: it destroys
   the record of what the crashed process actually did, which is exactly the
   evidence an incident review needs. Instead a `rollback` event is APPENDED,
   naming the offset range it orphans and why. `effective_events()` then
   reconstructs the logical stream by honouring those markers, while the file
   on disk keeps the full history.

   This is the transaction-log pattern the session's references point at
   (LakeFS / Iceberg / Delta): the log is the truth, and a rollback is another
   entry in it.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .hashing import GENESIS, chain

# Event types that carry a `global_step` and participate in rollback.
STEP_EVENTS = {"consume", "optimizer_step"}


class Ledger:
    """An append-only, hash-chained JSONL log."""

    def __init__(self, path: Path | str, run_id: str = "", branch: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.branch = branch
        self._events: List[dict] = []
        self._head = GENESIS
        if self.path.exists():
            self._load()

    # -- io ----------------------------------------------------------------
    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._events.append(json.loads(line))
        if self._events:
            self._head = self._events[-1]["event_hash"]

    RESERVED = ("seq", "type", "run_id", "branch", "ts", "prev_hash", "event_hash")

    def append(self, event_type: str, **fields) -> dict:
        # A caller passing e.g. branch= would collide with the envelope. The
        # envelope wins: those fields identify the log, not the event.
        payload = {k: v for k, v in fields.items() if k not in self.RESERVED}
        body = dict(
            seq=len(self._events),
            type=event_type,
            run_id=self.run_id,
            branch=self.branch,
            ts=round(time.time(), 6),
            **payload,
        )
        # The hash covers the body only; the hash field cannot hash itself.
        event_hash = chain(self._head, body)
        event = dict(body, prev_hash=self._head, event_hash=event_hash)
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())          # a crash must not lose the last event
        self._events.append(event)
        self._head = event_hash
        return event

    # -- state -------------------------------------------------------------
    @property
    def offset(self) -> int:
        """Number of events written. A checkpoint stores this."""
        return len(self._events)

    @property
    def head_hash(self) -> str:
        return self._head

    def hash_at(self, offset: int) -> str:
        """Chain hash after `offset` events -- what a checkpoint recorded."""
        if offset <= 0:
            return GENESIS
        if offset > len(self._events):
            raise IndexError(f"ledger has {len(self._events)} events, asked for {offset}")
        return self._events[offset - 1]["event_hash"]

    def all_events(self) -> List[dict]:
        """Raw history, rollbacks included. This is what is on disk."""
        return list(self._events)

    def events(self, event_type: str = None) -> List[dict]:
        ev = self.effective_events()
        return [e for e in ev if event_type is None or e["type"] == event_type]

    # -- rollback semantics ------------------------------------------------
    def effective_events(self) -> List[dict]:
        """The logical stream: history with orphaned ranges removed.

        Walks forward. A `rollback` event with `to_offset = k` discards every
        event whose `seq` is >= k and < the rollback's own seq. The rollback
        marker itself stays visible, because "we rolled back here" is part of
        the run's history.
        """
        out: List[dict] = []
        for ev in self._events:
            if ev["type"] == "rollback":
                to_offset = ev["to_offset"]
                out = [e for e in out if e["seq"] < to_offset or e["type"] == "rollback"]
            out.append(ev)
        return out

    def rollback(self, to_offset: int, reason: str, orphaned_steps: List[int]) -> dict:
        return self.append(
            "rollback",
            from_offset=self.offset,
            to_offset=to_offset,
            reason=reason,
            orphaned_steps=orphaned_steps,
            orphaned_events=self.offset - to_offset,
            head_at_rollback=self.head_hash,
        )

    # -- integrity ---------------------------------------------------------
    def verify_chain(self) -> dict:
        """Recompute the whole chain from genesis."""
        prev = GENESIS
        for i, ev in enumerate(self._events):
            body = {k: v for k, v in ev.items() if k not in ("prev_hash", "event_hash")}
            expected = chain(prev, body)
            if ev.get("prev_hash") != prev:
                return dict(ok=False, broken_at=i, reason="prev_hash_mismatch",
                            n_events=len(self._events))
            if expected != ev.get("event_hash"):
                return dict(ok=False, broken_at=i, reason="event_hash_mismatch",
                            n_events=len(self._events))
            prev = expected
        return dict(ok=True, broken_at=None, reason=None,
                    n_events=len(self._events), head=prev)

    # -- queries the audit uses -------------------------------------------
    def consumed_steps(self) -> List[int]:
        return sorted({e["global_step"] for e in self.events("consume")})

    def events_for_step(self, step: int) -> List[dict]:
        return [e for e in self.events("consume") if e["global_step"] == step]

    def events_in_step_range(self, lo: int, hi: int) -> List[dict]:
        return [e for e in self.events("consume") if lo <= e["global_step"] < hi]

    def last_committed_step(self) -> int:
        steps = [e["global_step"] for e in self.events("optimizer_step")]
        return max(steps) if steps else -1


@dataclass
class LedgerSet:
    """The five logs a run keeps, opened together."""
    consumption: Ledger
    opus: Ledger
    learning: Ledger
    firewall: Ledger
    token_trace: Ledger

    def verify_all(self) -> Dict[str, dict]:
        return {
            "consumption": self.consumption.verify_chain(),
            "opus": self.opus.verify_chain(),
            "learning": self.learning.verify_chain(),
            "firewall": self.firewall.verify_chain(),
            "token_trace": self.token_trace.verify_chain(),
        }


def open_ledgers(ledger_dir: Path | str, run_id: str, branch: str) -> LedgerSet:
    d = Path(ledger_dir)
    return LedgerSet(
        consumption=Ledger(d / f"consumption_{branch}.jsonl", run_id, branch),
        opus=Ledger(d / f"opus_{branch}.jsonl", run_id, branch),
        learning=Ledger(d / f"learning_{branch}.jsonl", run_id, branch),
        firewall=Ledger(d / f"firewall_{branch}.jsonl", run_id, branch),
        token_trace=Ledger(d / f"token_trace_{branch}.jsonl", run_id, branch),
    )
