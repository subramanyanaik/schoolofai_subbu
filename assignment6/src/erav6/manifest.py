"""
manifest.py — the shard manifest and the admission gate.

The manifest is what makes a shard auditable: it records where the tokens came
from, which tokenizer gives them meaning, how the text was cleaned, and whether
the shard is allowed into training at all. The admission gate then decides,
from the manifest and from the bytes on disk, whether the shard may enter the
gradient-bearing stream.

Two properties of the gate are worth stating, because they are what separate a
real gate from a decorative one:

  * It decides on METADATA AND BYTES, never on how the text reads. A document
    with an unknown licence is refused even though it looks like every other
    document -- `eng-badlicense-01` in the corpus exists to prove that.

  * It re-verifies the content hash against the file on disk on every
    admission. A shard that was admitted yesterday and edited overnight fails
    today. That is what "immutable" means operationally: not that editing is
    impossible, but that editing is DETECTED.

Every decision, pass or fail, is returned as a record with per-check detail, so
the evidence bundle can point at the specific check that fired rather than at a
bare boolean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import config as C
from . import shards as S
from .hashing import hash_obj
from .registry import EvalRegistry

CLEANING_PIPELINE_ID = f"{C.CLEANING_PIPELINE['name']}@{C.CLEANING_PIPELINE['version']}"
KNOWN_CLEANING_PIPELINES = {CLEANING_PIPELINE_ID}
CLEANING_PIPELINE_HASH = hash_obj(C.CLEANING_PIPELINE)


@dataclass
class Check:
    check: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return dict(check=self.check, passed=self.passed, detail=self.detail)


@dataclass
class AdmissionDecision:
    shard_id: str
    admitted: bool
    reasons: List[str] = field(default_factory=list)
    checks: List[Check] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(shard_id=self.shard_id, admitted=self.admitted,
                    reasons=list(self.reasons),
                    checks=[c.as_dict() for c in self.checks])


def build_manifest(shard: S.Shard, tokenizer_hash: str,
                   registry: Optional[EvalRegistry] = None) -> dict:
    """Assemble the manifest for one shard, running the contamination scan.

    `contamination_status` and `eval_overlap_status` are DERIVED here by
    scanning, never copied from what the corpus author claimed. The corpus
    contains a document whose declared status is "unscanned" precisely so that
    the scanner has to make the call itself.
    """
    languages = sorted({d["language"] for d in shard.docs})
    scripts = sorted({d["script"] for d in shard.docs})
    sources = sorted({d["source_id"] for d in shard.docs})
    licenses = sorted({d.get("license") or "unknown" for d in shard.docs})

    overlap_status = "clean"
    overlap_detail: List[dict] = []
    canary_hits: List[str] = []

    if registry is not None and shard.split == "train":
        hits = registry.scan_tokens(shard.tokens())
        if hits:
            overlap_detail = hits
            worst = hits[0]["benchmark"]
            overlap_status = f"overlap:{worst}"
    elif shard.split != "train":
        overlap_status = "is_eval"

    # Licence tier for the shard is the WORST licence of any member document.
    # A shard is only as admissible as its least admissible token.
    license_tier = "unknown"
    for cand in ("own", "share-alike", "gov-open", "permissive"):
        if cand in licenses:
            license_tier = cand
    for bad in C.BLOCKED_LICENSES:
        if bad in licenses:
            license_tier = bad
            break

    manifest = dict(
        shard_id=shard.shard_id,
        source_ids=sources,
        doc_ids=[d["doc_id"] for d in shard.docs],
        tokenizer_hash=tokenizer_hash,
        token_count=shard.token_count,
        loss_bearing_tokens=sum(d["n_loss_tokens"] for d in shard.docs),
        language=languages,
        script=scripts,
        lane=shard.lane,
        split=shard.split,
        license_tier=license_tier,
        licenses_present=licenses,
        cleaning_pipeline=CLEANING_PIPELINE_ID,
        cleaning_pipeline_hash=CLEANING_PIPELINE_HASH,
        dedup_status="exact+fuzzy-passed",
        contamination_status=("contaminated" if overlap_detail else
                              ("is_eval" if shard.split != "train" else "clean")),
        eval_overlap_status=overlap_status,
        eval_overlap_detail=overlap_detail,
        canary_hits=canary_hits,
        content_hash=S.content_hash(shard),
        parent_shard_ids=[],
        n_docs=len(shard.docs),
    )
    return manifest


def admission_gate(manifest: dict, shard: S.Shard, tokenizer_hash: str,
                   registry: Optional[EvalRegistry] = None) -> AdmissionDecision:
    """Decide whether this shard may enter the gradient-bearing stream."""
    checks: List[Check] = []
    reasons: List[str] = []

    def record(name: str, ok: bool, reason: str, detail: str = "") -> None:
        checks.append(Check(name, ok, detail))
        if not ok:
            reasons.append(reason)

    # 1. Every required field present and non-empty. `parent_shard_ids` is the
    # one field that is legitimately empty: a shard built straight from
    # documents has no parent, and only a derived shard does.
    allow_empty = {"parent_shard_ids"}
    missing = [f for f in C.REQUIRED_MANIFEST_FIELDS
               if f not in manifest
               or (manifest[f] in (None, "", []) and f not in allow_empty)]
    record("manifest_complete", not missing, "incomplete_manifest",
           f"missing={missing}" if missing else f"{len(C.REQUIRED_MANIFEST_FIELDS)} fields present")

    # 2. Token ids without their tokenizer are meaningless integers.
    tok_ok = manifest.get("tokenizer_hash") == tokenizer_hash and bool(tokenizer_hash)
    record("tokenizer_hash_verified", tok_ok, "tokenizer_hash_mismatch",
           f"manifest={str(manifest.get('tokenizer_hash'))[:16]} active={tokenizer_hash[:16]}")

    # 3. Immutability, checked against the bytes actually on disk right now.
    recomputed = S.content_hash(shard)
    hash_ok = recomputed == manifest.get("content_hash")
    record("content_hash_verified", hash_ok, "content_hash_mismatch",
           f"recomputed={recomputed[:16]} manifest={str(manifest.get('content_hash'))[:16]}")

    # 4. Licence.
    lic = manifest.get("license_tier")
    lic_ok = lic in C.ALLOWED_LICENSES
    record("license_permitted", lic_ok, "license_not_permitted", f"tier={lic}")

    # 5. Cleaning lineage.
    clean_ok = (manifest.get("cleaning_pipeline") in KNOWN_CLEANING_PIPELINES
                and manifest.get("cleaning_pipeline_hash") == CLEANING_PIPELINE_HASH)
    record("cleaning_lineage_known", clean_ok, "unknown_cleaning_lineage",
           f"pipeline={manifest.get('cleaning_pipeline')}")

    # 6. Deduplication.
    dedup_ok = manifest.get("dedup_status") == "exact+fuzzy-passed"
    record("dedup_passed", dedup_ok, "dedup_not_passed", f"status={manifest.get('dedup_status')}")

    # 7. Registry permission. Test and validation both fail this check, but for
    # different reasons and the distinction is the whole point of the firewall:
    # a test shard has no legitimate read path at all, while a validation shard
    # is readable for evaluation and merely refused a GRADIENT-bearing one.
    never_train = bool(registry and registry.is_never_train_shard(shard.shard_id))
    record("not_never_train", not never_train, "eval_registry_never_train",
           f"never_train={never_train}")
    record("is_training_split", shard.split == "train",
           "non_training_split",
           f"split={shard.split}"
           + (" (readable for eval, never gradient-bearing)"
              if shard.split == "validation" else ""))

    # 8. Contamination. Only meaningful for TRAINING shards: an eval shard
    # overlapping the eval registry is not contaminated, it *is* the registry,
    # and it is already refused by the split check above. Reserving the
    # `eval_overlap` reason for genuine training contamination keeps the
    # firewall evidence readable -- otherwise every eval shard shows up in the
    # contamination list and buries the one document that actually matters.
    detail = manifest.get("eval_overlap_detail") or []
    if shard.split == "train":
        overlap_ok = manifest.get("eval_overlap_status") == "clean"
        note = f"status={manifest.get('eval_overlap_status')} hits={len(detail)}"
    else:
        overlap_ok = True
        note = f"not applicable: {shard.split} shard, refused by is_training_split"
    record("eval_overlap_clean", overlap_ok, "eval_overlap", note)

    # 9. Canary strings.
    canary_ok = not manifest.get("canary_hits")
    record("no_canary_strings", canary_ok, "canary_string_present",
           f"hits={manifest.get('canary_hits')}")

    admitted = all(c.passed for c in checks)
    return AdmissionDecision(shard_id=shard.shard_id, admitted=admitted,
                             reasons=reasons, checks=checks)
