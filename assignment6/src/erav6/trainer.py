"""
trainer.py — the loop that turns the planned stream into gradients and records.

Everything the other modules built comes together here, in one order that
matters:

    build batch -> record OPUS decisions -> probe BEFORE -> forward/backward
    per microbatch (one consume event each) -> optimizer step -> probe AFTER
    -> learning ledger -> token trace -> checkpoint

The probes bracket the optimizer step because that is the only placement that
measures what the step did. Probing once per epoch, or comparing against a
running average, would confound the effect of this batch with everything else
that moved.

Crash injection is a real process death (`os._exit`), not an exception. An
exception unwinds the stack, runs `finally` blocks, flushes buffers and closes
files -- which is precisely the cleanup a real crash does not do. If the
recovery path is only ever tested against a polite exception, it has not been
tested. The ledger is fsynced on every append for the same reason.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from . import DATALOADER_VERSION
from . import checkpoint as CK
from . import config as C
from . import dataloader as DL
from . import learning as LRN
from . import model as MD
from . import opus as OP
from . import packing as PK
from .catalog import Catalog
from .ledger import LedgerSet
from .mixture import Timeline
from .perf import PerfMeter

CRASH_EXIT_CODE = 137


def lr_at(step: int, profile: dict) -> float:
    """Linear warmup then cosine decay to a tenth of peak."""
    peak = profile["lr"]
    warm = max(1, profile["warmup_steps"])
    total = profile["total_steps"]
    if step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


@dataclass
class StepResult:
    step: int
    stage: str
    loss: float
    grad_norm: float
    lr: float
    batch_content_hash: str
    n_loss_tokens: int
    n_positions: int
    utilization: float


class Trainer:
    def __init__(self, cat: Catalog, timeline: Timeline, profile: dict,
                 profile_name: str, branch: str, ledgers: LedgerSet,
                 ckpt_dir: Path, device: torch.device, run_id: str,
                 data_seed: int, log: Callable = None, meter: PerfMeter = None):
        self.cat = cat
        self.timeline = timeline
        self.profile = profile
        self.profile_name = profile_name
        self.branch = branch
        self.led = ledgers
        self.ckpt_dir = Path(ckpt_dir)
        self.device = device
        self.run_id = run_id
        self.data_seed = data_seed
        self.log = log or (lambda *a, **k: None)
        self.meter = meter or PerfMeter()

        torch.manual_seed(C.MODEL_SEED)
        self.model = MD.TinyTransformer(
            vocab_size=cat.tokenizer.vocab_size,
            d_model=profile["d_model"], n_layer=profile["n_layer"],
            n_head=profile["n_head"], d_ff=profile["d_ff"],
            max_seq_len=profile["seq_len"], dropout=profile["dropout"],
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=profile["lr"], betas=(0.9, 0.95),
                                           weight_decay=0.01)
        self.selector = OP.OpusSelector()
        self.loader = DL.DataLoader(cat, timeline, profile, branch, data_seed,
                                    self.selector, meter=self.meter)
        self.probes = LRN.ProbeSet(cat, profile["seq_len"], profile["probe_tokens"])
        self.learning = LRN.LearningLedger(cat, cat.tokenizer, profile["total_steps"])

        self.last_checkpoint_id = "none"
        self.tokens_consumed = 0
        self.step_results: List[StepResult] = []

    # -- branch bookkeeping ------------------------------------------------
    def open_branch(self, parent_branch: str = None, parent_step: int = None,
                    parent_ledger_hash: str = None, note: str = "") -> dict:
        return self.led.consumption.append(
            "branch_open",
            branch=self.branch,
            parent_branch=parent_branch,
            parent_step=parent_step,
            parent_ledger_hash=parent_ledger_hash,
            data_seed=self.data_seed,
            tokenizer_hash=self.cat.tokenizer_hash,
            catalog_set_hash=self.cat.set_hash(),
            dataloader_version=DATALOADER_VERSION,
            profile=self.profile_name,
            total_steps=self.profile["total_steps"],
            seqs_per_step=self.loader.seqs_per_step,
            seq_len=self.profile["seq_len"],
            note=note,
        )

    # -- one step ----------------------------------------------------------
    def train_step(self, step: int) -> StepResult:
        prof = self.profile
        stage = self.timeline.at(step).stage

        with self.meter.timer("dataload"):
            batch = self.loader.build_step(step, self.model, self.device,
                                           self.last_checkpoint_id)
        # OPUS scoring happens inside build_step; attribute its cost separately
        # so the throughput report can show what the selector costs.
        self.meter.add("candidates_scored", len(batch.decisions))
        for d in batch.decisions:
            if d.status == OP.REJECTED:
                self.meter.add("candidates_rejected")
            elif d.status == OP.DEFERRED:
                self.meter.add("candidates_deferred")
            else:
                self.meter.add("candidates_accepted")

        for d in batch.decisions:
            self.led.opus.append("opus_decision", global_step=step, **d.as_dict())

        for sub in batch.substitutions:
            self.led.consumption.append("lane_substitution", global_step=step, **sub)
        for sc in batch.scarcity:
            self.led.consumption.append("scarcity_fallback", **sc)

        shard_ids = sorted({m["shard_id"] for s in batch.sequences for m in s.members})
        with self.meter.timer("probe"):
            before = self.probes.evaluate(shard_ids, self.model, self.device)

        lr = lr_at(step, prof)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        total_loss, total_loss_tokens = 0.0, 0
        per_seq_losses: Dict[int, List[float]] = {}
        seq_index = 0

        for mb in batch.microbatches:
            with self.meter.timer("compute"):
                t = self.model.stack(mb.sequences, self.device)
                mean_loss, per_token = self.model.loss(
                    t["input_ids"], t["labels"], t["loss_mask"],
                    t["position_ids"], t["segment_ids"])
                # Scale so gradient accumulation averages rather than sums.
                (mean_loss / len(batch.microbatches)).backward()

            mask = t["loss_mask"]
            n_tok = int(mask.sum().item())
            total_loss += float(mean_loss.item()) * n_tok
            total_loss_tokens += n_tok
            for i in range(len(mb.sequences)):
                per_seq_losses[seq_index + i] = per_token[i].detach().cpu().tolist()
            seq_index += len(mb.sequences)

            self._record_consume(step, stage, mb, batch)
            self.meter.add("microbatches")

        with self.meter.timer("compute"):
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), prof["grad_clip"]).item())
            self.optimizer.step()

        with self.meter.timer("probe"):
            after = self.probes.evaluate(shard_ids, self.model, self.device)

        mean_loss_val = total_loss / max(1, total_loss_tokens)
        self.tokens_consumed += batch.n_real_tokens()

        self._record_learning(step, stage, batch, before, after, grad_norm,
                              per_seq_losses)

        result = StepResult(
            step=step, stage=stage, loss=mean_loss_val, grad_norm=grad_norm, lr=lr,
            batch_content_hash=batch.content_hash(),
            n_loss_tokens=batch.n_loss_tokens(),
            n_positions=batch.n_positions(),
            utilization=batch.utilization(),
        )
        self.step_results.append(result)

        self.led.consumption.append(
            "optimizer_step",
            global_step=step,
            batch_id=batch.batch_id(),
            batch_content_hash=result.batch_content_hash,
            curriculum_stage=stage,
            loss=round(mean_loss_val, 6),
            grad_norm=round(grad_norm, 6),
            lr=round(lr, 8),
            n_microbatches=len(batch.microbatches),
            n_sequences=len(batch.sequences),
            n_positions=result.n_positions,
            n_real_tokens=batch.n_real_tokens(),
            n_loss_tokens=result.n_loss_tokens,
            packing_utilization=round(result.utilization, 6),
            lane_counts=batch.lane_counts,
            floor_required=batch.floor_required,
            floor_delivered=batch.floor_delivered,
            tokens_consumed_total=self.tokens_consumed,
            checkpoint_id=self.last_checkpoint_id,
        )

        self.meter.add("steps")
        self.meter.add("sequences", len(batch.sequences))
        self.meter.add("positions", result.n_positions)
        self.meter.add("real_tokens", batch.n_real_tokens())
        self.meter.add("loss_tokens", result.n_loss_tokens)
        self.meter.add("accepted_loss_tokens", result.n_loss_tokens)
        self.meter.add("shard_reads", self.loader.shard_reads)
        self.meter.add("cache_hits", self.loader.cache_hits)
        self.loader.shard_reads = 0
        self.loader.cache_hits = 0
        return result

    def _record_consume(self, step: int, stage: str, mb: DL.Microbatch,
                        batch: DL.StepBatch) -> None:
        seq_records = []
        for s in mb.sequences:
            isolate = s.policy in PK.ISOLATING_POLICIES
            seq_records.append(dict(
                lane=s.lane,
                policy=s.policy,
                span_ids=s.provenance(),
                doc_ids=sorted({m["doc_id"] for m in s.members}),
                shard_ids=sorted({m["shard_id"] for m in s.members}),
                members=[dict(shard_id=m["shard_id"], doc_id=m["doc_id"],
                              shard_start=m["shard_start"], length=m["length"],
                              offset=m["offset"], segment_id=m["segment_id"])
                         for m in s.members],
                loss_mask_hash=s.loss_mask_hash(),
                content_hash=s.content_hash(),
                n_real_tokens=s.n_real_tokens,
                n_loss_tokens=s.n_loss_tokens,
                n_segments=len({x for x in s.segment_ids if x}),
                utilization=round(s.utilization, 6),
                n_truncated=s.n_truncated,
                boundary_crossings=s.boundary_crossings,
                loss_across_boundary=s.loss_across_boundary,
                attention_policy=("block_diagonal_causal_by_segment" if isolate
                                  else "causal_full_window"),
                position_policy="per_segment_restart" if isolate else "contiguous",
            ))

        self.led.consumption.append(
            "consume",
            global_step=step,
            checkpoint_id=self.last_checkpoint_id,
            rank=mb.rank,
            microbatch_id=mb.microbatch_id,
            batch_id=mb.batch_id,
            sequences=seq_records,
            span_ids=mb.span_ids(),
            shard_ids=mb.shard_ids(),
            doc_ids=mb.doc_ids(),
            loss_mask_hash=mb.loss_mask_hash(),
            content_hash=mb.content_hash(),
            mixture_lanes={s.lane: sum(1 for x in mb.sequences if x.lane == s.lane)
                           for s in mb.sequences},
            curriculum_stage=stage,
            tokenizer_hash=self.cat.tokenizer_hash,
            tokenizer_version=self.cat.tokenizer.version,
            dataloader_version=DATALOADER_VERSION,
            opus_decision_ids=[d.candidate_id for d in batch.decisions
                               if d.status in (OP.ACCEPTED, OP.FLOOR_OVERRIDE)],
        )

    def _record_learning(self, step: int, stage: str, batch: DL.StepBatch,
                         before: Dict[str, float], after: Dict[str, float],
                         grad_norm: float,
                         per_seq_losses: Dict[int, List[float]]) -> None:
        score_by_shard: Dict[str, List[float]] = {}
        for d in batch.decisions:
            if d.status in (OP.ACCEPTED, OP.FLOOR_OVERRIDE):
                for sid in d.shard_ids:
                    score_by_shard.setdefault(sid, []).append(d.score)

        per_shard_tokens: Dict[str, int] = {}
        per_shard_loss: Dict[str, float] = {}
        for idx, s in enumerate(batch.sequences):
            losses = per_seq_losses.get(idx, [])
            for m in s.members:
                sid = m["shard_id"]
                lo, hi = m["offset"], m["offset"] + m["length"]
                tok = sum(s.loss_mask[lo:hi])
                val = sum(losses[i] for i in range(lo, min(hi, len(losses)))
                          if s.loss_mask[i])
                per_shard_tokens[sid] = per_shard_tokens.get(sid, 0) + tok
                per_shard_loss[sid] = per_shard_loss.get(sid, 0.0) + val

        for sid, ntok in sorted(per_shard_tokens.items()):
            shard = self.cat.shards[sid]
            scores = score_by_shard.get(sid, [0.0])
            epoch = self.loader.state.lane_epoch.get(shard.lane, 0)
            rec = self.learning.record_exposure(
                shard_id=sid, lane=shard.lane, step=step, stage=stage,
                n_loss_tokens=ntok, sum_loss=per_shard_loss.get(sid, 0.0),
                grad_norm=grad_norm,
                opus_score=sum(scores) / len(scores),
                repeated_pass=epoch + 1)
            delta = None
            if sid in before and sid in after:
                delta = self.learning.record_delta(sid, before[sid], after[sid])

            self.led.learning.append(
                "shard_exposure", global_step=step, shard_id=sid,
                lane=shard.lane, curriculum_stage=stage,
                model_phase=LRN.model_phase(stage, step, self.profile["total_steps"]),
                loss_bearing_tokens=ntok,
                mean_token_loss=round(per_shard_loss.get(sid, 0.0) / max(1, ntok), 6),
                probe_loss_before=round(before[sid], 6) if sid in before else None,
                probe_loss_after=round(after[sid], 6) if sid in after else None,
                probe_loss_delta=round(delta, 6) if delta is not None else None,
                grad_norm=round(grad_norm, 6),
                mean_opus_score=round(sum(scores) / len(scores), 6),
                repeated_pass=epoch + 1,
                exposures_so_far=rec.exposures,
                tokens_consumed_when_seen=self.tokens_consumed,
                usefulness_so_far=rec.classify(),
            )

        for lane in sorted(batch.lane_counts):
            vals = [per_shard_loss[s] / max(1, per_shard_tokens[s])
                    for s in per_shard_tokens
                    if self.cat.shards[s].lane == lane and per_shard_tokens[s]]
            if vals:
                self.learning.record_lane_loss(lane, sum(vals) / len(vals))

        # Token-level trace, at the configured cadence.
        if step % self.profile["token_trace_every"] == 0 and batch.sequences:
            seq = batch.sequences[0]
            losses = per_seq_losses.get(0, [])
            if losses:
                score = next((d.score for d in batch.decisions
                              if d.status in (OP.ACCEPTED, OP.FLOOR_OVERRIDE)), 0.0)
                records = self.learning.token_trace(
                    seq, losses, step, stage,
                    checkpoint_before=self.last_checkpoint_id,
                    checkpoint_after="pending",
                    opus_score=score,
                    repeated_pass=self.loader.state.lane_epoch.get(seq.lane, 0) + 1,
                    tokens_consumed=self.tokens_consumed)
                self.led.token_trace.append(
                    "token_trace", global_step=step, lane=seq.lane,
                    sequence_content_hash=seq.content_hash(),
                    n_records=len(records), records=records)

    # -- checkpointing -----------------------------------------------------
    def save_checkpoint(self, next_step: int) -> dict:
        rec = CK.save(
            self.ckpt_dir, branch=self.branch, step=next_step,
            model=self.model, optimizer=self.optimizer,
            loader_state=self.loader.state_dict(),
            ledger_offset=self.led.consumption.offset,
            ledger_head_hash=self.led.consumption.head_hash,
            data_seed=self.data_seed,
            tokenizer_hash=self.cat.tokenizer_hash,
            catalog_set_hash=self.cat.set_hash(),
            run_id=self.run_id, profile_name=self.profile_name,
            extra=dict(tokens_consumed=self.tokens_consumed,
                       last_loss=round(self.step_results[-1].loss, 6)
                       if self.step_results else None),
        )
        self.last_checkpoint_id = rec["checkpoint_id"]
        self.led.consumption.append(
            "checkpoint", global_step=next_step,
            checkpoint_id=rec["checkpoint_id"],
            ledger_offset=rec["ledger_offset"],
            ledger_head_hash=rec["ledger_head_hash"],
            loader_state_hash=rec["loader_state_hash"],
            record_hash=rec["record_hash"],
            tokens_consumed=self.tokens_consumed)
        # A [PASS] marker rather than a bare event: the assignment names
        # `checkpoint_saved` as one of the markers run.log must carry, and the
        # claim it makes is checkable -- the checkpoint is only sound if its
        # recorded ledger head matches the ledger's actual head at that offset.
        binding_ok = (self.led.consumption.hash_at(rec["ledger_offset"])
                      == rec["ledger_head_hash"])
        if hasattr(self.log, "check"):
            self.log.check("checkpoint_saved", binding_ok,
                           checkpoint_id=rec["checkpoint_id"],
                           next_step=next_step,
                           ledger_offset=rec["ledger_offset"],
                           ledger_head=rec["ledger_head_hash"][:16],
                           loader_state_hash=rec["loader_state_hash"][:16],
                           binding_verified=binding_ok)
        else:
            self.log("checkpoint_saved", checkpoint_id=rec["checkpoint_id"],
                     next_step=next_step, ledger_offset=rec["ledger_offset"],
                     ledger_head=rec["ledger_head_hash"][:16])
        return rec

    def restore_checkpoint(self, record: dict) -> None:
        CK.restore(self.ckpt_dir, record, self.model, self.optimizer,
                   ledger=self.led.consumption, device=self.device)
        self.loader.load_state_dict(record["loader_state"])
        self.last_checkpoint_id = record["checkpoint_id"]
        self.tokens_consumed = record.get("tokens_consumed", 0)

    # -- driving -----------------------------------------------------------
    def run(self, from_step: int, to_step: int,
            crash_at: Optional[int] = None) -> List[StepResult]:
        out: List[StepResult] = []
        for step in range(from_step, to_step):
            if crash_at is not None and step == crash_at:
                self.led.consumption.append(
                    "crash", global_step=step, reason="injected_fault",
                    detail="process terminated mid-step by os._exit; no unwinding, "
                           "no flush, no cleanup",
                    exit_code=CRASH_EXIT_CODE,
                    last_checkpoint_id=self.last_checkpoint_id)
                self.log("crash_simulated", global_step=step,
                         exit_code=CRASH_EXIT_CODE,
                         last_checkpoint_id=self.last_checkpoint_id)
                # Not an exception: a real process death. `finally` blocks do
                # not run, buffers are not flushed, and the recovery path has to
                # cope with whatever the ledger managed to fsync.
                os._exit(CRASH_EXIT_CODE)

            res = self.train_step(step)
            self.log("step", global_step=step, stage=res.stage,
                     loss=round(res.loss, 4), grad_norm=round(res.grad_norm, 4),
                     util=round(res.utilization, 3),
                     loss_tokens=res.n_loss_tokens,
                     batch_hash=res.batch_content_hash[:16])
            out.append(res)

            if (step + 1) % self.profile["ckpt_every"] == 0:
                self.save_checkpoint(step + 1)
        return out
