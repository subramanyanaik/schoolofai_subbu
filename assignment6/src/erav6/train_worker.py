"""
train_worker.py — one training process. Runs, or dies, on its own.

This is a separate OS process on purpose. The crash phase launches it with
`--crash-at`, and it terminates via `os._exit`, which produces a genuine
non-zero exit status with no unwinding and no buffer flush. The resume phase
launches a FRESH process that knows nothing except what is on disk: the
checkpoint, the ledger, and the shards. If the recovery path depended on
anything held in memory by the parent, it would fail here, visibly.

Modes:
    train    start (or continue) a branch from step 0
    resume   restore the latest checkpoint of a branch, roll the ledger back to
             its recorded offset, and continue
    fork     restore a specific checkpoint of a parent branch onto a NEW branch
             with a new data seed, recording the divergence point

Usage (the demo drives this; it is not meant to be run by hand):
    python -m erav6.train_worker --mode train --until 24 --crash-at 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import catalog as CATALOG
from . import checkpoint as CK
from . import config as C
from . import mixture as MIX
from . import model as MD
from . import trainer as TR
from .ledger import open_ledgers
from .perf import PerfMeter
from .runlog import RunLog


def build_context(profile_name: str, log):
    prof = C.PROFILES[profile_name]
    seqs_per_step = prof["n_ranks"] * prof["microbatch"] * prof["grad_accum"]
    cat = CATALOG.build_catalog(log=None)
    tl = MIX.compile_timeline(cat, prof["total_steps"], seqs_per_step, prof["seq_len"])
    return prof, cat, tl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "resume", "fork"], required=True)
    ap.add_argument("--branch", default=C.MAIN_BRANCH)
    ap.add_argument("--parent-branch", default=C.MAIN_BRANCH)
    ap.add_argument("--from-step", type=int, default=0)
    ap.add_argument("--until", type=int, default=None)
    ap.add_argument("--crash-at", type=int, default=None)
    ap.add_argument("--fork-at", type=int, default=None)
    ap.add_argument("--profile", default=C.DEFAULT_PROFILE)
    ap.add_argument("--data-seed", type=int, default=C.DATA_SEED)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--artifacts", default=str(C.ARTIFACTS))
    ap.add_argument("--out", default=None, help="where to write this phase's summary")
    args = ap.parse_args(argv)

    artifacts = Path(args.artifacts)
    log = RunLog(artifacts / "run.log", phase=f"{args.mode}:{args.branch}")
    prof, cat, tl = build_context(args.profile, log)
    until = args.until if args.until is not None else prof["total_steps"]

    device = MD.resolve_device(args.device)
    ledgers = open_ledgers(artifacts / "ledgers", C.RUN_ID, args.branch)
    meter = PerfMeter()

    trainer = TR.Trainer(
        cat=cat, timeline=tl, profile=prof, profile_name=args.profile,
        branch=args.branch, ledgers=ledgers,
        ckpt_dir=artifacts / "checkpoints", device=device,
        run_id=C.RUN_ID, data_seed=args.data_seed, log=log, meter=meter)

    start = args.from_step

    # ---------------- mode: train ----------------------------------------
    if args.mode == "train":
        if ledgers.consumption.offset == 0:
            trainer.open_branch(note="initial branch of the run")
        log.event("training_started", branch=args.branch, from_step=start,
                  until=until, device=str(device),
                  model_params=trainer.model.n_params,
                  crash_at=args.crash_at if args.crash_at is not None else "none")

    # ---------------- mode: resume ---------------------------------------
    elif args.mode == "resume":
        rec = CK.latest_record(artifacts / "checkpoints", args.branch)
        if rec is None:
            log.check("resume_checkpoint_found", False, branch=args.branch)
            return 2
        start = rec["global_step"]

        # Everything the dead process recorded after this checkpoint never
        # reached the weights. Orphan it explicitly -- append a rollback marker
        # rather than truncating, so the crashed process's record survives.
        offset_now = ledgers.consumption.offset
        orphan_steps = sorted({
            e["global_step"] for e in ledgers.consumption.all_events()
            if e["type"] in ("consume", "optimizer_step")
            and e["seq"] >= rec["ledger_offset"]
        })
        rb = ledgers.consumption.rollback(
            to_offset=rec["ledger_offset"],
            reason="crash_recovery",
            orphaned_steps=orphan_steps)
        log.event("ledger_rolled_back", to_offset=rec["ledger_offset"],
                  from_offset=offset_now, orphaned_events=offset_now - rec["ledger_offset"],
                  orphaned_steps=orphan_steps)

        # The binding check lives inside restore_checkpoint: it compares the
        # ledger's chain hash at the recorded offset against the one the
        # checkpoint stored, and refuses to load if they disagree.
        trainer.restore_checkpoint(rec)
        log.check("checkpoint_ledger_binding_verified", True,
                  checkpoint_id=rec["checkpoint_id"],
                  ledger_offset=rec["ledger_offset"],
                  ledger_head=rec["ledger_head_hash"][:16])
        log.event("run_resumed", branch=args.branch, checkpoint_id=rec["checkpoint_id"],
                  next_step=start, tokens_consumed=trainer.tokens_consumed)

    # ---------------- mode: fork -----------------------------------------
    else:
        fork_at = args.fork_at if args.fork_at is not None else prof["fork_from_step"]
        rec = CK.load_record(artifacts / "checkpoints", args.parent_branch, fork_at)
        parent_led = open_ledgers(artifacts / "ledgers", C.RUN_ID, args.parent_branch)

        # A fork restores a parent checkpoint but writes to its OWN ledger, so
        # the binding check would compare against the wrong log. The parent
        # binding is verified explicitly instead, and recorded in the branch
        # open event as the divergence point.
        parent_hash_ok = (parent_led.consumption.hash_at(rec["ledger_offset"])
                          == rec["ledger_head_hash"])
        CK.restore(artifacts / "checkpoints", rec, trainer.model,
                   trainer.optimizer, ledger=None, device=device,
                   verify_binding=False)
        trainer.loader.load_state_dict(rec["loader_state"])
        trainer.last_checkpoint_id = rec["checkpoint_id"]
        trainer.tokens_consumed = rec.get("tokens_consumed", 0)
        # The fork's own branch name is part of every shuffle seed, so from
        # here the two streams diverge by construction, reproducibly.
        trainer.loader.state.branch = args.branch
        trainer.loader.state.data_seed = args.data_seed
        trainer.loader._order_cache.clear()

        trainer.open_branch(parent_branch=args.parent_branch, parent_step=fork_at,
                            parent_ledger_hash=rec["ledger_head_hash"],
                            note="deliberate fork: new data branch from a shared checkpoint")
        log.check("fork_parent_binding_verified", parent_hash_ok,
                  parent_branch=args.parent_branch, fork_at=fork_at,
                  parent_ledger_hash=rec["ledger_head_hash"][:16])
        log.event("branch_forked", branch=args.branch,
                  parent_branch=args.parent_branch, parent_step=fork_at,
                  new_data_seed=args.data_seed)
        start = fork_at

    results = trainer.run(start, until, crash_at=args.crash_at)

    log.event("training_finished", branch=args.branch, steps_run=len(results),
              first_step=results[0].step if results else None,
              last_step=results[-1].step if results else None,
              final_loss=round(results[-1].loss, 6) if results else None)

    summary = dict(
        mode=args.mode, branch=args.branch, from_step=start, until=until,
        device=str(device), n_steps=len(results),
        model_params=trainer.model.n_params,
        steps=[dict(step=r.step, stage=r.stage, loss=round(r.loss, 6),
                    grad_norm=round(r.grad_norm, 6), lr=r.lr,
                    batch_content_hash=r.batch_content_hash,
                    n_loss_tokens=r.n_loss_tokens, n_positions=r.n_positions,
                    utilization=round(r.utilization, 6)) for r in results],
        learning_report=trainer.learning.report(),
        loader_state=trainer.loader.state_dict(),
        scarcity=trainer.loader.state.scarcity,
        ledger_offset=ledgers.consumption.offset,
        ledger_head=ledgers.consumption.head_hash,
        performance=meter.report(device=str(device)),
    )
    out_path = Path(args.out) if args.out else (C.WORK / f"phase_{args.mode}_{args.branch}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
