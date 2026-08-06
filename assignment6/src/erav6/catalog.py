"""
catalog.py — corpus in, admitted shard pool out.

This is the seam between "data that exists" and "data the training stream is
allowed to draw from". It runs the whole admission path in order:

    documents -> tokenize (frozen tokenizer)
              -> register validation/test docs in the eval firewall
              -> write immutable shards
              -> build manifests (contamination scanned, not declared)
              -> admission gate
              -> admitted pool, indexed by lane

Nothing downstream may read a shard that is not in `admitted`. The mixture
scheduler is handed `by_lane`, which is derived from the admitted set only, so
a blocked shard is not merely flagged -- it is absent from the structure the
loader draws from, and there is no code path that reaches it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import config as C
from . import manifest as M
from . import shards as S
from . import tokenizer as T
from .hashing import canonical_json, merkle
from .registry import EvalRegistry, build_registry

Logger = Optional[Callable[..., None]]


@dataclass
class Catalog:
    tokenizer: T.Tokenizer
    registry: EvalRegistry
    shards: Dict[str, S.Shard] = field(default_factory=dict)
    manifests: Dict[str, dict] = field(default_factory=dict)
    decisions: List[M.AdmissionDecision] = field(default_factory=list)
    admitted: List[str] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)
    by_lane: Dict[str, List[str]] = field(default_factory=dict)
    docs_by_id: Dict[str, S.TokenizedDoc] = field(default_factory=dict)

    @property
    def tokenizer_hash(self) -> str:
        return self.tokenizer.hash

    def shard(self, shard_id: str) -> S.Shard:
        if shard_id not in self.admitted:
            # Defence in depth: even if a caller reaches past `by_lane`, a
            # blocked shard cannot be handed out for training.
            raise PermissionError(f"{shard_id} is not admitted; refusing to serve it")
        return self.shards[shard_id]

    def lane_tokens(self, lane: str) -> int:
        return sum(self.shards[s].token_count for s in self.by_lane.get(lane, []))

    def lane_loss_tokens(self, lane: str) -> int:
        return sum(self.manifests[s]["loss_bearing_tokens"] for s in self.by_lane.get(lane, []))

    def set_hash(self) -> str:
        """One hash covering the whole admitted pool, for the checkpoint record."""
        return merkle([self.manifests[s]["content_hash"] for s in sorted(self.admitted)])


def build_catalog(log: Logger = None, shard_dir: Path | str = None) -> Catalog:
    say = log or (lambda *a, **k: None)

    tok = T.load()
    say("tokenizer_loaded", tokenizer_hash=tok.hash, vocab_size=tok.vocab_size)

    raw_docs = S.load_documents()
    tdocs = [S.tokenize_document(d, tok) for d in raw_docs]
    say("documents_tokenized", n_documents=len(tdocs),
        n_tokens=sum(d.n_tokens for d in tdocs))

    # The firewall is built BEFORE any shard is admitted. A contamination scan
    # that runs after admission is a post-mortem, not a gate.
    registry = build_registry(tdocs)
    say("eval_registry_built", n_entries=len(registry.entries),
        n_fingerprints=registry.n_fingerprints, benchmarks=registry.benchmarks())

    shard_list = S.build_shards(tdocs, out_dir=shard_dir, tokenizer_hash=tok.hash)
    say("shards_created", n_shards=len(shard_list),
        n_tokens=sum(s.token_count for s in shard_list))

    cat = Catalog(tokenizer=tok, registry=registry)
    cat.docs_by_id = {d.doc_id: d for d in tdocs}

    # Test shards are flagged never-train before any manifest is built, so the
    # gate sees the flag rather than inferring it.
    for sh in shard_list:
        cat.shards[sh.shard_id] = sh
        if sh.split == "test":
            registry.mark_never_train_shard(sh.shard_id)

    for sh in shard_list:
        man = M.build_manifest(sh, tok.hash, registry)
        # Canary sweep on decoded text: cheap, and a canary hit means the
        # breach happened upstream of us.
        canaries = registry.scan_text(tok.decode(sh.tokens()))
        if canaries and sh.split == "train":
            man["canary_hits"] = canaries
            man["contamination_status"] = "canary"
        cat.manifests[sh.shard_id] = man

        decision = M.admission_gate(man, sh, tok.hash, registry)
        cat.decisions.append(decision)

        if decision.admitted:
            cat.admitted.append(sh.shard_id)
            cat.by_lane.setdefault(sh.lane, []).append(sh.shard_id)
        else:
            cat.blocked.append(sh.shard_id)
            say("shard_blocked", shard_id=sh.shard_id, lane=sh.lane,
                split=sh.split, reasons=decision.reasons)

    for lane in cat.by_lane:
        cat.by_lane[lane].sort()

    say("manifests_validated", n_admitted=len(cat.admitted),
        n_blocked=len(cat.blocked), set_hash=cat.set_hash()[:16])
    return cat


def write_manifests(cat: Catalog, out_dir: Path | str = None) -> Path:
    """Emit the manifest half of the graded bundle."""
    out_dir = Path(out_dir or C.ART_MANIFESTS)
    out_dir.mkdir(parents=True, exist_ok=True)

    for shard_id, man in cat.manifests.items():
        with open(out_dir / f"{shard_id}.json", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(canonical_json(man))

    index = dict(
        tokenizer_hash=cat.tokenizer_hash,
        tokenizer_name=cat.tokenizer.name,
        tokenizer_version=cat.tokenizer.version,
        vocab_size=cat.tokenizer.vocab_size,
        cleaning_pipeline=M.CLEANING_PIPELINE_ID,
        cleaning_pipeline_hash=M.CLEANING_PIPELINE_HASH,
        n_shards=len(cat.shards),
        n_admitted=len(cat.admitted),
        n_blocked=len(cat.blocked),
        admitted=sorted(cat.admitted),
        blocked=sorted(cat.blocked),
        admitted_set_hash=cat.set_hash(),
        lanes={lane: dict(shards=ids,
                          tokens=cat.lane_tokens(lane),
                          loss_bearing_tokens=cat.lane_loss_tokens(lane))
               for lane, ids in sorted(cat.by_lane.items())},
        decisions=[d.as_dict() for d in cat.decisions],
    )
    path = out_dir / "_index.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        import json
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")

    cat.registry.save(out_dir / "_eval_registry.json")
    return path
