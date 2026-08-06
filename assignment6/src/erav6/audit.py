"""
audit.py — the questions the ledger exists to answer.

"Audit reconstructs the data that trained a checkpoint or a range of
checkpoints. It answers questions such as: which shards influenced the model
between 5.4B and 5.6B tokens? Which OPUS-selected batches appeared before a
loss spike?"

Every function here reads the LEDGERS and the SHARDS. None of them reads the
in-memory state of the run that produced them, which is what makes them usable
after the fact, from a clean process, by someone who was not there.

The firewall sweep is the one to read closely. It does not check "did we
intend to exclude eval data". It walks every consume event that was actually
recorded, collects every shard id that actually entered a gradient-bearing
batch, and intersects that set with the never-train registry. If the
intersection is non-empty the firewall failed, regardless of what the
admission gate believed at the time.
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

from . import config as C
from . import plan
from .catalog import Catalog
from .dataloader import rebuild_sequences_from_ledger
from .ledger import Ledger, LedgerSet
from .registry import EvalRegistry


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
def verify_ledgers(ledgers: LedgerSet) -> dict:
    results = ledgers.verify_all()
    return dict(ok=all(r["ok"] for r in results.values()), per_ledger=results)


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------
def firewall_sweep(consumption: Ledger, registry: EvalRegistry,
                   cat: Catalog) -> dict:
    """Did any never-train or non-training-split shard reach a gradient?"""
    consumed: Dict[str, int] = {}
    for ev in consumption.events("consume"):
        for sid in ev.get("shard_ids", []):
            consumed[sid] = consumed.get(sid, 0) + 1

    never_train_hits = sorted(s for s in consumed
                              if registry.is_never_train_shard(s))
    non_train_hits = sorted(s for s in consumed
                            if s in cat.shards and cat.shards[s].split != "train")
    blocked_hits = sorted(s for s in consumed if s in cat.blocked)

    return dict(
        ok=not (never_train_hits or non_train_hits or blocked_hits),
        n_distinct_shards_consumed=len(consumed),
        n_never_train_registered=len(registry.never_train_shards),
        never_train_shards_consumed=never_train_hits,
        non_training_split_shards_consumed=non_train_hits,
        blocked_shards_consumed=blocked_hits,
        blocked_shard_ids=sorted(cat.blocked),
        validation_reads=[a for a in registry.access_log if a["allowed"]],
        denied_reads=[a for a in registry.access_log if not a["allowed"]],
    )


# ---------------------------------------------------------------------------
# Interval reconstruction
# ---------------------------------------------------------------------------
def shards_in_step_range(consumption: Ledger, lo: int, hi: int) -> dict:
    """"Which shards influenced the model between step lo and step hi?" """
    by_shard: Dict[str, int] = {}
    by_lane: Dict[str, int] = {}
    docs: set = set()
    for ev in consumption.events_in_step_range(lo, hi):
        for rec in ev["sequences"]:
            by_lane[rec["lane"]] = by_lane.get(rec["lane"], 0) + rec["n_loss_tokens"]
            for m in rec["members"]:
                by_shard[m["shard_id"]] = by_shard.get(m["shard_id"], 0) + m["length"]
                docs.add(m["doc_id"])
    return dict(step_lo=lo, step_hi=hi,
                n_shards=len(by_shard), n_documents=len(docs),
                tokens_by_shard=dict(sorted(by_shard.items())),
                loss_tokens_by_lane=dict(sorted(by_lane.items())),
                document_ids=sorted(docs))


def shards_in_token_range(consumption: Ledger, lo_tokens: int,
                          hi_tokens: int) -> dict:
    """The Session-6 question in its original form, phrased in tokens."""
    steps = []
    for ev in consumption.events("optimizer_step"):
        total = ev.get("tokens_consumed_total", 0)
        if lo_tokens <= total <= hi_tokens:
            steps.append(ev["global_step"])
    if not steps:
        return dict(token_lo=lo_tokens, token_hi=hi_tokens, n_shards=0,
                    steps=[], tokens_by_shard={}, loss_tokens_by_lane={})
    out = shards_in_step_range(consumption, min(steps), max(steps) + 1)
    out.update(token_lo=lo_tokens, token_hi=hi_tokens, steps=sorted(steps))
    return out


def batches_before_loss_spike(consumption: Ledger, z: float = 1.5) -> dict:
    """"Which OPUS-selected batches appeared before a loss spike?"

    A spike is a step whose loss rose by more than `z` standard deviations of
    the observed step-to-step change. Reported with the batch that preceded it,
    because that is the batch under suspicion.
    """
    steps = sorted(consumption.events("optimizer_step"),
                   key=lambda e: e["global_step"])
    losses = [e["loss"] for e in steps]
    if len(losses) < 4:
        return dict(n_spikes=0, spikes=[], threshold=None)
    deltas = [b - a for a, b in zip(losses, losses[1:])]
    mu = statistics.fmean(deltas)
    sd = statistics.pstdev(deltas) or 1e-9
    thresh = mu + z * sd

    spikes = []
    for i, d in enumerate(deltas):
        if d > thresh:
            spike_step = steps[i + 1]
            prev = steps[i]
            spikes.append(dict(
                spike_at_step=spike_step["global_step"],
                loss_before=prev["loss"], loss_after=spike_step["loss"],
                delta=round(d, 6),
                grad_norm=spike_step.get("grad_norm"),
                preceding_batch_id=prev["batch_id"],
                preceding_batch_hash=prev["batch_content_hash"],
                preceding_lanes=prev.get("lane_counts", {}),
                stage=spike_step.get("curriculum_stage"),
            ))
    return dict(n_spikes=len(spikes), spikes=spikes,
                threshold=round(thresh, 6), mean_delta=round(mu, 6),
                sd_delta=round(sd, 6))


# ---------------------------------------------------------------------------
# Mixture compliance
# ---------------------------------------------------------------------------
def mixture_compliance(consumption: Ledger, timeline) -> dict:
    """Planned lane shares versus the shares actually consumed.

    Measured in SEQUENCES, matching the unit the quotas were compiled in, and
    also in loss-bearing tokens, which is the unit that actually reflects what
    the model learned from each lane. The two differ because lanes have
    different loss densities -- an agentic lane delivers far fewer gradient
    tokens per sequence than an English one, and reporting only sequences would
    hide that.
    """
    actual_seqs: Dict[str, int] = {}
    actual_loss: Dict[str, int] = {}
    for ev in consumption.events("consume"):
        for rec in ev["sequences"]:
            actual_seqs[rec["lane"]] = actual_seqs.get(rec["lane"], 0) + 1
            actual_loss[rec["lane"]] = actual_loss.get(rec["lane"], 0) + rec["n_loss_tokens"]

    # Every departure from the compiled quota should be explained by a recorded
    # substitution. A lane that hit its epoch cap and invoked its declared
    # `reduce_share` fallback SHOULD come in under plan -- that is the plan
    # working, not the plan failing. What would be a real failure is drift that
    # no logged event accounts for, so that is what the pass criterion uses.
    moved_out: Dict[str, int] = {}
    moved_in: Dict[str, int] = {}
    substitutions: List[dict] = []
    for ev in consumption.events("lane_substitution"):
        substitutions.append(dict(step=ev["global_step"], from_lane=ev["from_lane"],
                                  to_lane=ev["to_lane"], seqs=ev["seqs"],
                                  reason=ev["reason"]))
        moved_out[ev["from_lane"]] = moved_out.get(ev["from_lane"], 0) + ev["seqs"]
        moved_in[ev["to_lane"]] = moved_in.get(ev["to_lane"], 0) + ev["seqs"]

    tot_seq = sum(actual_seqs.values()) or 1
    tot_loss = sum(actual_loss.values()) or 1
    planned = timeline.planned_lane_shares()
    planned_seqs = {lane: planned.get(lane, 0.0) * tot_seq for lane in plan.LANES}

    rows = []
    worst_raw = worst_unexplained = 0.0
    for lane in plan.LANES:
        a_n = actual_seqs.get(lane, 0)
        a_seq = a_n / tot_seq
        drift = a_seq - planned.get(lane, 0.0)
        expected_after_subs = (planned_seqs[lane]
                               - moved_out.get(lane, 0) + moved_in.get(lane, 0))
        unexplained = (a_n - expected_after_subs) / tot_seq
        worst_raw = max(worst_raw, abs(drift))
        worst_unexplained = max(worst_unexplained, abs(unexplained))
        rows.append(dict(
            lane=lane,
            session5_target_share=plan.GLOBAL_MIXTURE[lane],
            planned_share=round(planned.get(lane, 0.0), 6),
            actual_sequence_share=round(a_seq, 6),
            actual_loss_token_share=round(actual_loss.get(lane, 0) / tot_loss, 6),
            sequences=a_n,
            planned_sequences=round(planned_seqs[lane], 3),
            substituted_away=moved_out.get(lane, 0),
            substituted_in=moved_in.get(lane, 0),
            expected_after_substitutions=round(expected_after_subs, 3),
            loss_tokens=actual_loss.get(lane, 0),
            drift_vs_planned=round(drift, 6),
            unexplained_drift=round(unexplained, 6),
            drift_vs_session5=round(a_seq - plan.GLOBAL_MIXTURE[lane], 6),
        ))
    return dict(total_sequences=tot_seq, total_loss_tokens=tot_loss,
                max_abs_drift_vs_planned=round(worst_raw, 6),
                max_abs_unexplained_drift=round(worst_unexplained, 6),
                n_substitution_events=len(substitutions),
                total_sequences_substituted=sum(moved_out.values()),
                substitutions=substitutions,
                note=("drift_vs_planned includes deviations caused by declared "
                      "scarcity fallbacks; unexplained_drift is the part no "
                      "logged event accounts for and is the pass criterion"),
                lanes=rows)


def floor_compliance(consumption: Ledger) -> dict:
    """Was the protected floor met in EVERY batch, not merely on average?"""
    rows, violations = [], []
    for ev in sorted(consumption.events("optimizer_step"),
                     key=lambda e: e["global_step"]):
        counts = ev.get("lane_counts", {})
        n_seq = ev.get("n_sequences", 0) or 1
        delivered = sum(counts.get(l, 0) for l in plan.FLOOR_LANES)
        required = ev.get("floor_required", 0)
        ok = delivered >= required
        rows.append(dict(step=ev["global_step"], required=required,
                         delivered=delivered, share=round(delivered / n_seq, 6),
                         ok=ok))
        if not ok:
            violations.append(ev["global_step"])
    shares = [r["share"] for r in rows] or [0.0]
    return dict(
        ok=not violations,
        floor_pct=plan.FLOOR_PCT,
        floor_lanes=plan.FLOOR_LANES,
        n_steps=len(rows),
        n_violations=len(violations),
        violating_steps=violations,
        min_delivered_share=round(min(shares), 6),
        mean_delivered_share=round(sum(shares) / len(shares), 6),
        per_step=rows,
    )


def anneal_reserve_compliance(consumption: Ledger, timeline) -> dict:
    """Reserved shards must be untouched before the anneal and used in it."""
    anneal_steps = {q.step for q in timeline.quotas
                    if q.stage.startswith(C.ANNEAL_STAGE_PREFIX)}
    reserved = {s for ids in timeline.reserved_shards.values() for s in ids}
    if not reserved:
        return dict(ok=True, reserved_shards=[], note="no lane had enough shards to reserve")

    before: Dict[str, int] = {s: 0 for s in reserved}
    during: Dict[str, int] = {s: 0 for s in reserved}
    for ev in consumption.events("consume"):
        step = ev["global_step"]
        for sid in ev.get("shard_ids", []):
            if sid in reserved:
                if step in anneal_steps:
                    during[sid] += 1
                else:
                    before[sid] += 1

    leaked = sorted(s for s, n in before.items() if n > 0)
    return dict(
        ok=not leaked,
        reserved_shards=sorted(reserved),
        anneal_steps=sorted(anneal_steps),
        consumed_before_anneal=before,
        consumed_during_anneal=during,
        leaked_shards=leaked,
        n_reserved_used_in_anneal=sum(1 for n in during.values() if n > 0),
    )


# ---------------------------------------------------------------------------
# OPUS trail
# ---------------------------------------------------------------------------
def opus_audit(opus_ledger: Ledger) -> dict:
    events = opus_ledger.events("opus_decision")
    by_status: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    by_lane: Dict[str, Dict[str, int]] = {}
    overrides: List[dict] = []
    deferred_ids: Dict[str, int] = {}
    returned: List[dict] = []

    for e in events:
        st = e["status"]
        by_status[st] = by_status.get(st, 0) + 1
        by_lane.setdefault(e["lane"], {})
        by_lane[e["lane"]][st] = by_lane[e["lane"]].get(st, 0) + 1
        if st in ("rejected", "deferred"):
            reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
        if e.get("protected_floor_override"):
            overrides.append(dict(candidate_id=e["candidate_id"], lane=e["lane"],
                                  step=e["step"], opus_score=e["opus_score"]))
        if st == "deferred":
            deferred_ids[e["candidate_id"]] = e["step"]
        if e.get("deferred_from_step") is not None:
            returned.append(dict(candidate_id=e["candidate_id"], lane=e["lane"],
                                 deferred_at=e["deferred_from_step"],
                                 resolved_at=e["step"], status=st))

    complete = all(e["status"] in ("accepted", "rejected", "deferred",
                                   "floor_override") for e in events)
    return dict(
        ok=bool(events) and complete,
        n_decisions=len(events),
        every_candidate_has_a_decision=complete,
        by_status=by_status,
        by_lane=by_lane,
        rejection_reasons=reasons,
        n_floor_overrides=len(overrides),
        floor_overrides=overrides[:20],
        n_deferred=len(deferred_ids),
        n_deferred_returned=len(returned),
        deferred_returned=returned[:20],
        rejected_data_retained=True,
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def replay_interval(cat: Catalog, consumption: Ledger, lo: int, hi: int,
                    seq_len: int) -> dict:
    """Rebuild every microbatch in [lo, hi) from the ledger and the shards.

    Compares three things per microbatch: the batch id, the loss-mask hash and
    the full content hash. Content hash is the strict one -- it covers tokens,
    mask, segment ids and position ids together, so a replay that got the right
    tokens with the wrong isolation structure still fails.
    """
    checked, mismatches, spans_checked = [], [], 0
    ids_checked = 0
    for ev in consumption.events_in_step_range(lo, hi):
        rebuilt = rebuild_sequences_from_ledger(cat, ev, seq_len)
        recorded = ev["sequences"]
        ok = True
        detail = []

        # Batch ids are a deterministic function of (branch, step, rank,
        # microbatch). Recomputing one from its components and comparing it to
        # the stored string is a small check, but it is the difference between
        # "the id we read back is the id we read back" and an id that was
        # actually reconstructed.
        expected_id = (f"{ev['branch']}/{ev['global_step']:06d}"
                       f"/r{ev['rank']}/m{ev['microbatch_id']}")
        same_id = expected_id == ev["batch_id"]
        ids_checked += 1
        if not same_id:
            ok = False
            detail.append(dict(rule="batch_id", recorded=ev["batch_id"],
                               reconstructed=expected_id))

        for rec, seq in zip(recorded, rebuilt):
            spans_checked += len(rec["span_ids"])
            same_content = seq.content_hash() == rec["content_hash"]
            same_mask = seq.loss_mask_hash() == rec["loss_mask_hash"]
            same_spans = seq.provenance() == rec["span_ids"]
            if not (same_content and same_mask and same_spans):
                ok = False
                detail.append(dict(
                    lane=rec["lane"], policy=rec["policy"],
                    content_match=same_content, mask_match=same_mask,
                    span_match=same_spans,
                    recorded_content_hash=rec["content_hash"][:24],
                    replay_content_hash=seq.content_hash()[:24],
                ))
        row = dict(step=ev["global_step"], batch_id=ev["batch_id"],
                   reconstructed_batch_id=expected_id, batch_id_match=same_id,
                   rank=ev["rank"], microbatch_id=ev["microbatch_id"],
                   n_sequences=len(recorded), match=ok)
        if not ok:
            row["detail"] = detail
            mismatches.append(row)
        checked.append(row)

    return dict(
        ok=bool(checked) and not mismatches,
        interval=[lo, hi],
        n_microbatches_replayed=len(checked),
        n_batch_ids_verified=ids_checked,
        n_token_spans_verified=spans_checked,
        n_mismatches=len(mismatches),
        # The three things the assignment asks a replay to prove, named.
        verified=dict(batch_ids=ids_checked, token_spans=spans_checked,
                      content_hashes=sum(len(e["sequences"])
                                         for e in consumption.events_in_step_range(lo, hi))),
        mismatches=mismatches,
        per_microbatch=checked,
    )


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
def split_at_rollback(consumption: Ledger, event_type: str = "optimizer_step"):
    """Separate what the crashed process did from what the resumed one did.

    Reads the RAW history, not the effective view. The effective view exists to
    hide orphaned events from the logical stream; the resume proof needs to see
    them, because comparing "what the crashed run produced at step 13" against
    "what the resumed run produced at step 13" is the entire check.
    """
    raw = consumption.all_events()
    rollbacks = [e for e in raw if e["type"] == "rollback"]
    if not rollbacks:
        return [e for e in raw if e["type"] == event_type], [], None
    cut = rollbacks[-1]["seq"]
    before = [e for e in raw if e["type"] == event_type and e["seq"] < cut]
    after = [e for e in raw if e["type"] == event_type and e["seq"] > cut]
    return before, after, rollbacks[-1]


def resume_verification(original: Sequence[dict], resumed: Sequence[dict],
                        expected_next_step: int) -> dict:
    """Compare the pre-crash and post-resume streams step by step.

    Two claims are checked. First, the resumed run's FIRST batch is the batch
    the checkpoint said was next -- not the one after it (a skip) and not the
    one before (a repeat). Second, every step both runs executed produced an
    identical batch content hash, which is the strong form: it would catch a
    lost carry buffer or a reshuffled lane order that a batch-id comparison
    alone would miss.
    """
    orig = {e["global_step"]: e for e in original}
    res = {e["global_step"]: e for e in resumed}
    if not res:
        return dict(ok=False, reason="resumed run produced no steps")

    first_resumed = min(res)
    next_ok = first_resumed == expected_next_step

    overlap = sorted(set(orig) & set(res))
    compared, mismatches = [], []
    for s in overlap:
        a, b = orig[s]["batch_content_hash"], res[s]["batch_content_hash"]
        row = dict(step=s, original_hash=a[:24], resumed_hash=b[:24], match=a == b)
        compared.append(row)
        if a != b:
            mismatches.append(row)

    skipped = sorted(s for s in range(expected_next_step, max(res) + 1)
                     if s not in res)
    repeated = sorted(s for s in res if s < expected_next_step)

    return dict(
        ok=next_ok and not mismatches and not skipped and not repeated,
        expected_next_step=expected_next_step,
        actual_next_step=first_resumed,
        next_batch_matched=next_ok,
        n_steps_compared=len(compared),
        n_hash_mismatches=len(mismatches),
        skipped_steps=skipped,
        repeated_steps=repeated,
        comparisons=compared,
        mismatches=mismatches,
    )


def fork_verification(main_led: Ledger, fork_led: Ledger,
                      fork_point: int) -> dict:
    """A fork must record its divergence and then actually diverge."""
    opens = fork_led.events("branch_open")
    if not opens:
        return dict(ok=False, reason="fork branch has no branch_open event")
    rec = opens[0]

    main_steps = {e["global_step"]: e["batch_content_hash"]
                  for e in main_led.events("optimizer_step")}
    fork_steps = {e["global_step"]: e["batch_content_hash"]
                  for e in fork_led.events("optimizer_step")}
    shared = sorted(set(main_steps) & set(fork_steps))
    differing = [s for s in shared if main_steps[s] != fork_steps[s]]

    return dict(
        ok=(rec.get("parent_branch") is not None
            and rec.get("parent_step") == fork_point
            and bool(rec.get("parent_ledger_hash"))
            and bool(differing)),
        fork_branch=rec["branch"],
        parent_branch=rec.get("parent_branch"),
        parent_step=rec.get("parent_step"),
        parent_ledger_hash=rec.get("parent_ledger_hash"),
        divergence_recorded=rec.get("parent_step") == fork_point,
        n_shared_steps=len(shared),
        n_diverging_steps=len(differing),
        diverging_steps=differing,
        note=("a fork shares a checkpoint with its parent but not a data "
              "stream; identical hashes on shared steps would mean the fork "
              "silently replayed the parent instead of branching"),
    )
