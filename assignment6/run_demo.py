#!/usr/bin/env python3
"""
run_demo.py — the complete demonstration, in one command.

    python run_demo.py

Regenerates `submission_artifacts/` from scratch: run.log, evidence.json,
evidence.md, manifests/, ledgers/, checkpoints/ and performance.json. No manual
intervention, no network, no arguments required.

The training phases run as SUBPROCESSES rather than function calls. That is not
tidiness -- it is the point of the crash demonstration. The crash phase exits
via `os._exit(137)`: no stack unwinding, no `finally`, no flush. The resume
phase is a brand-new interpreter that knows only what is on disk. If recovery
secretly depended on live objects, it would fail here in a way an in-process
`try/except` would hide.

Phases:
     1  freeze check      tokenizer loaded and hash verified
     2  shards            corpus tokenized into immutable shards + manifests
     3  immutability      a mutated copy of a shard is rejected
     4  firewall          test shard blocked; contaminated training shard caught
     5  mixture           Session-5 curriculum compiled into per-step quotas
     6  packing           all six policies measured on real lane data
     7  train             branch `main` runs until it deliberately crashes
     8  resume            fresh process recovers and proves the next batch
     9  replay            an earlier interval is rebuilt from shards + ledger
    10  fork              a new branch diverges from an earlier checkpoint
    11  audit             firewall sweep, interval queries, chain verification
    12  performance       measured throughput and packing efficiency
    13  evidence          bundle re-derived from the artefacts, in its own process
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from erav6 import audit as AU                    # noqa: E402
from erav6 import catalog as CATALOG             # noqa: E402
from erav6 import config as C                    # noqa: E402
from erav6 import mixture as MIX                 # noqa: E402
from erav6 import packing as PK                  # noqa: E402
from erav6 import plan                           # noqa: E402
from erav6 import shards as SH                   # noqa: E402
from erav6 import tokenizer as TOK               # noqa: E402
from erav6.ledger import open_ledgers            # noqa: E402
from erav6.perf import PerfMeter, rejection_rate_by_lane   # noqa: E402
from erav6.runlog import RunLog                  # noqa: E402
from erav6.trainer import CRASH_EXIT_CODE        # noqa: E402


def clean(artifacts: Path, work: Path) -> None:
    """A demonstration that cannot be regenerated from scratch is not evidence."""
    for d in (artifacts, work):
        if d.exists():
            shutil.rmtree(d)
    for sub in ("manifests", "ledgers", "checkpoints"):
        (artifacts / sub).mkdir(parents=True, exist_ok=True)


def worker(log: RunLog, *args: str, expect_exit: int = 0) -> int:
    """Run a training phase as a real subprocess."""
    cmd = [sys.executable, "-m", "erav6.train_worker", *args]
    env_src = str(HERE / "src")
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = env_src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=str(HERE), env=env)
    log.event("subprocess_exit", cmd=" ".join(args), exit_code=proc.returncode,
              expected=expect_exit)
    return proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=C.DEFAULT_PROFILE, choices=sorted(C.PROFILES))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--artifacts", default=str(C.ARTIFACTS))
    ap.add_argument("--keep", action="store_true",
                    help="do not wipe previous artifacts first")
    args = ap.parse_args(argv)

    artifacts = Path(args.artifacts)
    prof = C.PROFILES[args.profile]
    t_start = time.perf_counter()

    if not args.keep:
        clean(artifacts, C.WORK)

    log = RunLog(artifacts / "run.log", phase="demo", append=args.keep)
    log.section("ERA V5 SESSION 6 - TRAINING DATA EXECUTION SYSTEM")
    log.info(f"profile={args.profile} device={args.device} "
             f"steps={prof['total_steps']} seq_len={prof['seq_len']} "
             f"global_batch={prof['n_ranks'] * prof['microbatch'] * prof['grad_accum']}")
    log.info(f"executing the Session-5 plan imported from assignment5/src/erav5: "
             f"{len(plan.LANES)} lanes, {len(plan.STAGE_NAMES)} curriculum stages, "
             f"protected floor {plan.FLOOR_PCT:.0%} native Indic")

    meter = PerfMeter()

    # -- 1. tokenizer ------------------------------------------------------
    log.section("PHASE 1 - FROZEN TOKENIZER")
    p1 = log.sub("tokenizer")
    with open(C.TOKENIZER_PATH, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    recomputed = TOK.tokenizer_hash(spec)
    tok = TOK.load()
    probe = "मानसून monsoon calc(expr=\"3*45\") தமிழ்"
    round_trip = tok.decode(tok.encode(probe)) == probe
    p1.event("tokenizer_loaded", name=tok.name, version=tok.version,
             vocab_size=tok.vocab_size, n_merges=len(tok.merges))
    p1.check("tokenizer_hash_verified", recomputed == tok.hash and round_trip,
             tokenizer_hash=recomputed[:32], multibyte_round_trip=round_trip,
             vocab_size=tok.vocab_size)

    # -- 2. shards and manifests -------------------------------------------
    log.section("PHASE 2 - SHARDS, MANIFESTS AND ADMISSION")
    p2 = log.sub("shards")
    with meter.timer("catalog"):
        cat = CATALOG.build_catalog(log=p2.event)
    CATALOG.write_manifests(cat, artifacts / "manifests")
    p2.event("shards_created", n_shards=len(cat.shards),
             n_tokens=sum(s.token_count for s in cat.shards.values()),
             shard_dir=str(C.SHARD_DIR))
    p2.event("manifests_validated", n_admitted=len(cat.admitted),
             n_blocked=len(cat.blocked),
             admitted_set_hash=cat.set_hash()[:16])
    p2.check("manifests_complete",
             all(set(C.REQUIRED_MANIFEST_FIELDS).issubset(m) for m in cat.manifests.values()),
             n_manifests=len(cat.manifests),
             required_fields=len(C.REQUIRED_MANIFEST_FIELDS))
    p2.check("all_shards_bound_to_tokenizer",
             all(m["tokenizer_hash"] == tok.hash for m in cat.manifests.values()),
             n_manifests=len(cat.manifests))

    # -- 3. immutability ----------------------------------------------------
    log.section("PHASE 3 - SHARD IMMUTABILITY")
    p3 = log.sub("immutability")
    victim_id = sorted(cat.admitted)[0]
    victim = cat.shards[victim_id]
    tamper_dir = C.WORK / "tamper"
    tamper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(victim.bin_path, tamper_dir / victim.bin_path.name)
    shutil.copy(victim.idx_path, tamper_dir / victim.idx_path.name)
    raw = bytearray((tamper_dir / victim.bin_path.name).read_bytes())
    raw[8] = (raw[8] + 1) % 256                     # flip one byte of one token
    (tamper_dir / victim.bin_path.name).write_bytes(bytes(raw))
    tampered = SH.load_shard(victim_id, tamper_dir)
    original_hash = cat.manifests[victim_id]["content_hash"]
    tampered_hash = SH.content_hash(tampered)
    detected = not SH.verify(tampered, original_hash)
    p3.event("shard_tamper_test", shard_id=victim_id, bytes_changed=1)
    p3.check("shard_immutability_enforced", detected, shard_id=victim_id,
             original_hash=original_hash[:24], tampered_hash=tampered_hash[:24],
             admission_would_reject=detected)

    # -- 4. evaluation firewall --------------------------------------------
    log.section("PHASE 4 - EVALUATION AND VALIDATION FIREWALL")
    p4 = log.sub("firewall")
    contaminated = [d for d in cat.decisions if "eval_overlap" in d.reasons]
    never_train = [d for d in cat.decisions if "eval_registry_never_train" in d.reasons]
    val_split = [d for d in cat.decisions if "non_training_split" in d.reasons]
    bad_license = [d for d in cat.decisions if "license_not_permitted" in d.reasons]

    p4.event("eval_registry", n_entries=len(cat.registry.entries),
             n_fingerprints=cat.registry.n_fingerprints,
             ngram=C.CONTAMINATION_NGRAM, benchmarks=cat.registry.benchmarks())
    for d in cat.decisions:
        if not d.admitted:
            p4.event("eval_shard_blocked" if any(
                r in d.reasons for r in ("eval_overlap", "eval_registry_never_train",
                                         "non_training_split"))
                else "shard_blocked",
                shard_id=d.shard_id, reasons=d.reasons)
    overlap_detail = []
    for d in contaminated:
        detail = cat.manifests[d.shard_id].get("eval_overlap_detail") or []
        if detail:
            overlap_detail = detail
            break
    p4.check("eval_shard_blocked",
             bool(contaminated) and bool(never_train),
             contamination_blocked=[d.shard_id for d in contaminated],
             never_train_blocked=[d.shard_id for d in never_train],
             validation_split_blocked=[d.shard_id for d in val_split],
             matched_ngrams=overlap_detail[0]["matched_ngrams"] if overlap_detail else 0,
             benchmark=overlap_detail[0]["benchmark"] if overlap_detail else "-")
    p4.check("license_gate_enforced", bool(bad_license),
             blocked=[d.shard_id for d in bad_license])

    # A validation shard may be READ for evaluation but never for a gradient.
    val_shards = [s for s in cat.shards.values() if s.split == "validation"]
    denied = False
    if val_shards:
        cat.registry.open_for_eval(val_shards[0].shard_id, "held_out_perplexity")
        try:
            cat.registry.open_for_eval(val_shards[0].shard_id, "training")
        except PermissionError:
            denied = True
    p4.check("validation_read_allowed_gradient_denied", denied,
             shard_id=val_shards[0].shard_id if val_shards else "-",
             eval_read="allowed", gradient_read="denied")

    # -- 5. mixture timeline ------------------------------------------------
    log.section("PHASE 5 - MIXTURE TIMELINE")
    p5 = log.sub("mixture")
    seqs_per_step = prof["n_ranks"] * prof["microbatch"] * prof["grad_accum"]
    tl = MIX.compile_timeline(cat, prof["total_steps"], seqs_per_step, prof["seq_len"])
    with open(artifacts / "manifests" / "_mixture_timeline.json", "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(tl.as_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    planned = tl.planned_lane_shares()
    drift = max(abs(planned[l] - plan.GLOBAL_MIXTURE[l]) for l in plan.LANES)
    over_cap = [f["lane"] for f in tl.feasibility if f["status"] != "ok"]
    p5.event("mixture_compiled", n_steps=tl.total_steps,
             seqs_per_step=seqs_per_step,
             stages=[a["stage"] for a in tl.stage_alloc],
             reserved_shards=tl.reserved_shards)
    for f in tl.feasibility:
        if f["status"] != "ok":
            p5.event("lane_scarcity", lane=f["lane"], status=f["status"],
                     epochs=f["epochs"], cap=f["epoch_cap_scaled"],
                     fallback=f["fallback"])
    p5.check("mixture_integrates_to_session5_plan", drift <= 0.05,
             max_drift=round(drift, 6), tolerance=0.05, n_lanes=len(plan.LANES))
    p5.check("protected_floor_satisfiable",
             all(q.floor_delivered >= q.floor_required for q in tl.quotas),
             floor_pct=plan.FLOOR_PCT, floor_lanes=plan.FLOOR_LANES,
             min_required=min(q.floor_required for q in tl.quotas))
    p5.check("scarcity_fallbacks_declared",
             all(f["fallback"] for f in tl.feasibility if f["status"] != "ok"),
             lanes_over_cap=over_cap)

    # -- 6. packing ---------------------------------------------------------
    log.section("PHASE 6 - PACKING POLICIES")
    p6 = log.sub("packing")
    policy_report = {}
    violations = 0
    for lane in ["english", "agentic", "reasoning", "code"]:
        pool = []
        for sid in cat.by_lane.get(lane, []):
            shard = cat.shards[sid]
            toks = shard.tokens()
            for d in shard.docs:
                loss = [0] * d["length"]
                for sp in d["spans"]:
                    if sp["loss"]:
                        lo = sp["start"] - d["start"]
                        for i in range(lo, lo + sp["length"]):
                            if 0 <= i < d["length"]:
                                loss[i] = 1
                pool.append(PK.Sample(sid, d["doc_id"], lane, d["start"], d["length"],
                                      toks[d["start"]:d["start"] + d["length"]],
                                      loss, d["spans"]))
        for pol in C.ALL_PACKING_POLICIES:
            res = PK.pack(pool, prof["seq_len"], pol, max_seqs=6, lane=lane)
            st = res.stats()
            policy_report[f"{lane}/{pol}"] = st
            if pol in PK.ISOLATING_POLICIES and st["total_loss_across_boundary"]:
                violations += 1
            p6.event("batches_packed", lane=lane, policy=pol,
                     sequences=st["n_sequences"],
                     utilization=st["mean_utilization"],
                     loss_density=st["mean_loss_density"],
                     truncated=st["total_truncated"],
                     boundary_crossings=st["total_boundary_crossings"],
                     loss_across_boundary=st["total_loss_across_boundary"])
    with open(artifacts / "manifests" / "_packing_report.json", "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(policy_report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    p6.check("packing_isolation_enforced", violations == 0,
             policies=len(C.ALL_PACKING_POLICIES),
             isolating_policies=sorted(PK.ISOLATING_POLICIES),
             loss_across_boundary_violations=violations)

    # -- 7. train until the crash ------------------------------------------
    log.section("PHASE 7 - TRAINING (WILL CRASH DELIBERATELY)")
    rc = worker(log, "--mode", "train", "--branch", C.MAIN_BRANCH,
                "--until", str(prof["total_steps"]),
                "--crash-at", str(prof["crash_at_step"]),
                "--profile", args.profile, "--device", args.device,
                "--artifacts", str(artifacts),
                "--out", str(C.WORK / "phase_train.json"),
                expect_exit=CRASH_EXIT_CODE)
    p7 = log.sub("crash")
    p7.check("crash_produced_nonzero_exit", rc == CRASH_EXIT_CODE,
             exit_code=rc, expected=CRASH_EXIT_CODE,
             crash_at_step=prof["crash_at_step"],
             mechanism="os._exit, no unwinding or flush")

    # -- 8. resume ----------------------------------------------------------
    log.section("PHASE 8 - RESUME FROM CHECKPOINT")
    rc = worker(log, "--mode", "resume", "--branch", C.MAIN_BRANCH,
                "--until", str(prof["total_steps"]),
                "--profile", args.profile, "--device", args.device,
                "--artifacts", str(artifacts),
                "--out", str(C.WORK / "phase_resume.json"))
    p8 = log.sub("resume")
    if rc != 0:
        p8.check("resume_process_completed", False, exit_code=rc)
        return 1

    main_led = open_ledgers(artifacts / "ledgers", C.RUN_ID, C.MAIN_BRANCH)
    before, after, rb = AU.split_at_rollback(main_led.consumption)
    from erav6 import checkpoint as CK
    resume_ckpt = None
    for p in sorted((artifacts / "checkpoints").glob(f"ckpt-{C.MAIN_BRANCH}-*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rb and rec["ledger_offset"] == rb["to_offset"]:
            resume_ckpt = rec
    expected_next = resume_ckpt["global_step"] if resume_ckpt else -1
    ver = AU.resume_verification(before, after, expected_next)

    p8.check("resume_next_batch_matched", ver["next_batch_matched"],
             expected_next_step=ver["expected_next_step"],
             actual_next_step=ver["actual_next_step"],
             checkpoint_id=resume_ckpt["checkpoint_id"] if resume_ckpt else "-")
    p8.check("resume_no_skipped_or_repeated_batches",
             not ver["skipped_steps"] and not ver["repeated_steps"],
             skipped=ver["skipped_steps"], repeated=ver["repeated_steps"],
             orphaned_by_rollback=rb["orphaned_steps"] if rb else [])
    p8.check("resume_batch_hashes_identical", ver["n_hash_mismatches"] == 0,
             steps_compared=ver["n_steps_compared"],
             mismatches=ver["n_hash_mismatches"])

    # -- 9. replay ----------------------------------------------------------
    log.section("PHASE 9 - REPLAY OF A HISTORICAL INTERVAL")
    p9 = log.sub("replay")
    lo, hi = prof["replay_interval"]
    with meter.timer("replay"):
        rep = AU.replay_interval(cat, main_led.consumption, lo, hi, prof["seq_len"])
    p9.event("historical_stream_replayed", interval=[lo, hi],
             microbatches=rep["n_microbatches_replayed"],
             token_spans=rep["n_token_spans_verified"])
    p9.check("replay_hash_matched", rep["ok"], interval=[lo, hi],
             microbatches_replayed=rep["n_microbatches_replayed"],
             token_spans_verified=rep["n_token_spans_verified"],
             mismatches=rep["n_mismatches"])

    # -- 10. fork -----------------------------------------------------------
    log.section("PHASE 10 - FORK FROM AN EARLIER CHECKPOINT")
    rc = worker(log, "--mode", "fork", "--branch", C.FORK_BRANCH,
                "--parent-branch", C.MAIN_BRANCH,
                "--fork-at", str(prof["fork_from_step"]),
                "--until", str(prof["fork_from_step"] + prof["fork_steps"]),
                "--data-seed", str(C.DATA_SEED + 991),
                "--profile", args.profile, "--device", args.device,
                "--artifacts", str(artifacts),
                "--out", str(C.WORK / "phase_fork.json"))
    p10 = log.sub("fork")
    if rc != 0:
        p10.check("fork_process_completed", False, exit_code=rc)
        return 1
    fork_led = open_ledgers(artifacts / "ledgers", C.RUN_ID, C.FORK_BRANCH)
    fk = AU.fork_verification(main_led.consumption, fork_led.consumption,
                              prof["fork_from_step"])
    p10.check("fork_divergence_recorded", fk["ok"],
              branch=fk.get("fork_branch"), parent=fk.get("parent_branch"),
              parent_step=fk.get("parent_step"),
              shared_steps=fk.get("n_shared_steps"),
              diverging_steps=fk.get("n_diverging_steps"))

    # -- 10b. protected-floor drill ----------------------------------------
    log.section("PHASE 10b - PROTECTED FLOOR DRILL (CONTROLLED COUNTERFACTUAL)")
    p10b = log.sub("floor_drill")
    p10b.info("the real run's proxy PREFERS Indic (it is the highest-loss "
              "content at this scale), so the override never fires there; this "
              "drill scores one identical candidate pool with the English-tuned "
              "proxy Session 5 warns about, with and without the floor")
    from erav6 import floor_drill as FD
    from erav6 import model as MD
    import torch
    final_ckpt = CK.latest_record(artifacts / "checkpoints", C.MAIN_BRANCH)
    device = MD.resolve_device(args.device)
    drill_model = MD.TinyTransformer(
        cat.tokenizer.vocab_size, prof["d_model"], prof["n_layer"],
        prof["n_head"], prof["d_ff"], prof["seq_len"], prof["dropout"]).to(device)
    blob = torch.load(artifacts / "checkpoints" / final_ckpt["weights_file"],
                      map_location=device, weights_only=False)
    drill_model.load_state_dict(blob["model"])
    drill = FD.run(cat, tl, prof, drill_model, device, step=prof["fork_from_step"])
    with open(artifacts / "ledgers" / "_floor_drill.json", "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(drill, fh, indent=2, sort_keys=True)
        fh.write("\n")
    p10b.event("protected_floor_drill", proxy=drill.get("proxy"),
               candidates=drill.get("n_candidates"),
               unprotected_native_indic=drill["unprotected"]["native_indic_sequences"],
               protected_native_indic=drill["protected"]["native_indic_sequences"],
               floor_required=drill.get("floor_required"))
    p10b.check("protected_floor_override_demonstrated",
               drill.get("floor_was_necessary", False)
               and drill["protected"]["n_floor_overrides"] > 0,
               unprotected_native_indic=drill["unprotected"]["native_indic_sequences"],
               protected_native_indic=drill["protected"]["native_indic_sequences"],
               floor_required=drill["floor_required"],
               n_overrides=drill["protected"]["n_floor_overrides"],
               mean_biased_score_native=drill["biased_score_gap"]["mean_native_indic"],
               mean_biased_score_other=drill["biased_score_gap"]["mean_other"])

    # -- 11. audit ----------------------------------------------------------
    log.section("PHASE 11 - AUDIT")
    p11 = log.sub("audit")
    chains = AU.verify_ledgers(main_led)
    fork_chains = AU.verify_ledgers(fork_led)
    sweep = AU.firewall_sweep(main_led.consumption, cat.registry, cat)
    comp = AU.mixture_compliance(main_led.consumption, tl)
    floor = AU.floor_compliance(main_led.consumption)
    reserve = AU.anneal_reserve_compliance(main_led.consumption, tl)
    trail = AU.opus_audit(main_led.opus)
    spikes = AU.batches_before_loss_spike(main_led.consumption)
    interval = AU.shards_in_step_range(main_led.consumption, lo, hi)
    tok_range = AU.shards_in_token_range(main_led.consumption, 0, 10_000)

    audit_report = dict(
        ledger_chains=chains, fork_ledger_chains=fork_chains,
        firewall=sweep, mixture_compliance=comp, floor_compliance=floor,
        anneal_reserve=reserve, opus=trail, loss_spikes=spikes,
        interval_query=interval, token_range_query=tok_range,
        resume=ver, replay=rep, fork=fk, protected_floor_drill=drill,
    )
    with open(artifacts / "ledgers" / "_audit_report.json", "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(audit_report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    p11.event("opus_decisions_recorded", n=trail["n_decisions"],
              by_status=trail["by_status"],
              floor_overrides=trail["n_floor_overrides"],
              deferred_returned=trail["n_deferred_returned"])
    p11.check("ledger_chain_intact", chains["ok"] and fork_chains["ok"],
              main=[k for k, v in chains["per_ledger"].items() if v["ok"]],
              n_events=main_led.consumption.offset)
    p11.check("no_eval_data_in_training", sweep["ok"],
              never_train_consumed=sweep["never_train_shards_consumed"],
              blocked_consumed=sweep["blocked_shards_consumed"],
              distinct_shards_consumed=sweep["n_distinct_shards_consumed"])
    # The pass criterion is UNEXPLAINED drift. A lane that hit its epoch cap
    # and invoked its declared fallback is supposed to come in under plan; what
    # would be a real failure is a deviation no logged event accounts for.
    p11.check("mixture_deviation_fully_attributed",
              comp["max_abs_unexplained_drift"] <= 0.02,
              max_unexplained_drift=comp["max_abs_unexplained_drift"],
              tolerance=0.02,
              max_raw_drift=comp["max_abs_drift_vs_planned"],
              substitution_events=comp["n_substitution_events"],
              sequences_substituted=comp["total_sequences_substituted"])
    p11.check("protected_floor_held_every_batch", floor["ok"],
              steps=floor["n_steps"], violations=floor["n_violations"],
              min_share=floor["min_delivered_share"])
    p11.check("anneal_reserve_respected", reserve["ok"],
              reserved=reserve.get("reserved_shards", []),
              leaked=reserve.get("leaked_shards", []))
    p11.check("opus_every_candidate_decided", trail["ok"],
              decisions=trail["n_decisions"],
              statuses=sorted(trail["by_status"]))
    p11.event("audit_completed", queries=["interval", "token_range", "loss_spikes",
                                          "firewall", "mixture", "floor", "reserve"],
              interval_shards=interval["n_shards"],
              interval_documents=interval["n_documents"],
              loss_spikes=spikes["n_spikes"])

    # -- 12. performance ----------------------------------------------------
    log.section("PHASE 12 - PERFORMANCE")
    p12 = log.sub("performance")
    train_sum = json.loads((C.WORK / "phase_train.json").read_text(encoding="utf-8")) \
        if (C.WORK / "phase_train.json").exists() else {}
    resume_sum = json.loads((C.WORK / "phase_resume.json").read_text(encoding="utf-8"))
    fork_sum = json.loads((C.WORK / "phase_fork.json").read_text(encoding="utf-8"))

    combined = PerfMeter()
    train_wall = 0.0
    for part in (train_sum, resume_sum, fork_sum):
        perf = part.get("performance", {})
        train_wall += perf.get("wall_clock_s", 0.0)
        for k, v in perf.get("timers_s", {}).items():
            combined.timers[k] = combined.timers.get(k, 0.0) + v
        for k, v in perf.get("timer_calls", {}).items():
            combined.calls[k] = combined.calls.get(k, 0) + v
        # Only the process-local counters are summed from the workers. The
        # token counters are taken from the ledger instead -- see below.
        for k in ("shard_reads", "cache_hits"):
            combined.counters[k] = combined.counters.get(k, 0.0) + \
                perf.get("raw_counters", {}).get(k, 0.0)

    # Token accounting comes from the LEDGERS, not from the worker summaries.
    # The crashed process died via os._exit and never wrote a summary, so
    # summing the summaries would silently omit the twelve steps it completed
    # before the crash -- an undercount of a third of the run. The ledger was
    # fsynced on every append, so it has them, and it is also the artefact a
    # grader can re-derive these numbers from.
    for led in (main_led, fork_led):
        for ev in led.consumption.events("optimizer_step"):
            combined.add("steps")
            combined.add("microbatches", ev.get("n_microbatches", 0))
            combined.add("sequences", ev.get("n_sequences", 0))
            combined.add("positions", ev.get("n_positions", 0))
            combined.add("real_tokens", ev.get("n_real_tokens", 0))
            combined.add("loss_tokens", ev.get("n_loss_tokens", 0))
            combined.add("accepted_loss_tokens", ev.get("n_loss_tokens", 0))
        for e in led.opus.events("opus_decision"):
            combined.add("candidates_scored")
            status = e["status"]
            if status == "rejected":
                combined.add("candidates_rejected")
            elif status == "deferred":
                combined.add("candidates_deferred")
            else:
                combined.add("candidates_accepted")

    # Wall clock is the whole demonstration, measured here, not summed.
    combined._t0 = t_start

    decisions = [e for e in main_led.opus.events("opus_decision")]
    report = combined.report(
        device=resume_sum.get("device", "unknown"),
        train_wall_s=train_wall,
        extra=dict(
            profile=args.profile,
            coverage=dict(
                token_counts_source="consumption ledgers (effective events, both branches)",
                timers_source="training worker subprocesses",
                wall_clock_source="the whole run_demo.py invocation",
                note=("the crashed worker contributes no timers because it was "
                      "terminated with os._exit and never wrote a summary; its "
                      "completed steps are still counted, because the ledger "
                      "was fsynced on every append"),
                branches=[C.MAIN_BRANCH, C.FORK_BRANCH],
            ),
            phases=dict(
                train=train_sum.get("performance", {}).get("time_s", {}),
                resume=resume_sum.get("performance", {}).get("time_s", {}),
                fork=fork_sum.get("performance", {}).get("time_s", {}),
            ),
            replay_latency_s=round(combined.timers.get("replay", 0.0), 6),
            catalog_build_s=round(meter.timers.get("catalog", 0.0), 6),
            rejection_rate_by_lane=rejection_rate_by_lane(decisions),
            packing_policy_comparison=policy_report,
            packing_utilization_from_ledger=round(
                sum(e.get("n_real_tokens", 0) for e in
                    main_led.consumption.events("optimizer_step"))
                / max(1, sum(e.get("n_positions", 0) for e in
                             main_led.consumption.events("optimizer_step"))), 6),
        ))
    # `replay` was timed on this process's meter, not the workers'.
    report["timers_s"]["replay"] = round(meter.timers.get("replay", 0.0), 6)
    with open(artifacts / "performance.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    thr = report["throughput"]
    eff = report["efficiency"]
    p12.event("performance_measured",
              raw_tokens_per_s=thr["raw_tokens_per_s"],
              useful_tokens_per_s=thr["useful_loss_bearing_tokens_per_s"],
              packing_utilization=eff["packing_utilization"],
              padding_waste=eff["padding_waste"],
              context_only=eff["context_only_fraction"],
              opus_acceptance=eff["opus_acceptance_rate"],
              cache_hit_rate=eff["shard_cache_hit_rate"])
    p12.check("throughput_measured",
              thr["useful_loss_bearing_tokens_per_s"] > 0
              and eff["packing_utilization"] > 0,
              useful_tokens_per_s=thr["useful_loss_bearing_tokens_per_s"],
              raw_tokens_per_s=thr["raw_tokens_per_s"],
              packing_utilization=eff["packing_utilization"])

    # -- 13. evidence -------------------------------------------------------
    log.section("PHASE 13 - EVIDENCE BUNDLE")
    p13 = log.sub("evidence")
    p13.info("regenerating evidence in a separate process that never imports "
             "the trainer, reading only the artefacts on disk")
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE / "src") + os.pathsep + env.get("PYTHONPATH", "")
    rc = subprocess.run(
        [sys.executable, "-c",
         "from erav6 import evidence; import sys; "
         f"b = evidence.generate(r'{artifacts}', '{args.profile}'); "
         "sys.exit(0 if b['summary']['all_passed'] else 3)"],
        cwd=str(HERE), env=env).returncode

    bundle = json.loads((artifacts / "evidence.json").read_text(encoding="utf-8"))
    s = bundle["summary"]
    for r in bundle["requirements"]:
        p13.event("requirement", key=r["key"], result=r["result"])
    p13.check("evidence_bundle_generated", s["all_passed"],
              passed=s["passed"], total=s["total"], failed=s["failed"])

    log.section("SUMMARY")
    log.info(f"requirements passed: {s['passed']}/{s['total']}")
    log.info(f"artifacts: {artifacts}")
    log.info(f"total wall clock: {time.perf_counter() - t_start:.1f}s")
    failed = [m["name"] for m in log.markers if m["result"] == "FAIL"]
    if failed:
        log.info(f"FAILED CHECKS: {failed}")
    return 0 if (s["all_passed"] and not failed) else 1


if __name__ == "__main__":
    sys.exit(main())
