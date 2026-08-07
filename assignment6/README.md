# ERA V5 — Session 6: Training Data Execution System

[![validate](https://github.com/subramanyanaik/schoolofai_subbu/actions/workflows/validate.yml/badge.svg)](https://github.com/subramanyanaik/schoolofai_subbu/actions/workflows/validate.yml)
![evidence](https://img.shields.io/badge/evidence-11%2F11%20PASS-brightgreen)
![tests](https://img.shields.io/badge/tests-97%2F97%20passing-brightgreen)
![runtime](https://img.shields.io/badge/one%20command-under%202%20min-blue)
![python](https://img.shields.io/badge/python-3.12-blue)

**Session-6 assignment.** The [Session-5 mixture plan](../assignment5/README.md) turned into an executed, ledgered, replayable data stream — and then deliberately crashed, resumed, replayed and forked to prove it can be reconstructed.

```bash
git clone https://github.com/subramanyanaik/schoolofai_subbu.git
cd schoolofai_subbu/assignment6
pip install torch                 # the only dependency
python run_demo.py                # regenerates all of submission_artifacts/
echo $?                           # 0 = all 11 requirements PASS
```

One command. No arguments, no network, no manual intervention.

Measured wall clock: **27s** on an SSD with CUDA available, **74–92s** from a fresh clone on a slower volume, **79s** forced CPU-only (`--device cpu`, which is what CI runs). The device is auto-selected and recorded in `performance.json`; nothing here needs a GPU.

| | |
|---|---|
| **Pipeline** | documents → shards → manifests → mixture → packing → batches → training → consumption ledger → learning ledger → checkpoint → **crash** → resume → replay → fork → audit |
| **Plan executed** | Session 5's, **imported** from `assignment5/src/erav5` — 11 lanes, 5 stages, 8% native-Indic floor |
| **Crash** | real process death, `os._exit(137)` — not a caught exception |
| **Resume** | next batch = step 12 as the checkpoint named; **4 batch content hashes identical** across the crash |
| **Replay** | 24 batch ids, **72 token spans**, every content hash — rebuilt from the shards, 0 mismatches |
| **Evidence** | **11/11 PASS**, re-derived from artefacts in a separate process |
| **Tests** | **97** invariant tests, stdlib `unittest`, no install needed |

---

## Contents

| § | Section |
|---|---|
| [1](#1-what-this-system-is) | What this system is |
| [2](#2-the-five-decisions-that-carry-the-weight) | The five decisions that carry the weight |
| [3](#3-architecture) | Architecture |
| [4](#4-shards-manifests-and-the-admission-gate) | Shards, manifests and the admission gate |
| [5](#5-the-evaluation-firewall) | The evaluation firewall |
| [6](#6-packing-masks-and-position-ids) | Packing, masks and position ids |
| [7](#7-mixture-floors-and-scarcity) | Mixture, floors and scarcity |
| [8](#8-opus-and-its-audit-trail) | OPUS and its audit trail |
| [9](#9-crash-resume-replay-fork) | Crash, resume, replay, fork |
| [10](#10-the-two-way-learning-ledger) | The two-way learning ledger |
| [11](#11-throughput) | Throughput |
| [12](#12-evidence-and-tests) | Evidence and tests |
| [13](#13-what-this-does-not-prove) | What this does not prove |
| [14](#14-repository-layout) | Repository layout |

---

## 1. What this system is

Session 5 produced a *plan*: 22% code, 8% native-Indic floor, five curriculum stages, an anneal reserve, an OPUS selector. A dataloader cannot consume a percentage. Session 6 is the machinery that turns that plan into the actual tensors the optimizer sees, and — more importantly — into a record complete enough to answer, months later:

> what did this checkpoint consume, why did it consume it, what did the model learn from it, and can the run be reconstructed?

The system is deliberately small: 512-token vocabulary, ~890K-parameter model, a 14,375-token corpus. The assignment is explicit that scale is not the point. Every design decision below is about *correctness, reproducibility and auditability*, which is what the grading is for.

**Session 6 executes Session 5's plan literally.** `src/erav6/plan.py` imports `assignment5/src/erav5` and reads the lane shares, stage mixes, protected floor and epoch caps out of it. It refuses to start if that package is missing. A Session-6 system carrying its own private copy of the mixture could drift out of agreement with the plan it claims to execute, and every "mixture compliance" claim downstream would then be measuring the wrong target.

---

## 2. The five decisions that carry the weight

### 2.1 The crash is a real process death

`run_demo.py` runs the training phases as **subprocesses**. The crash phase calls `os._exit(137)` mid-step: no stack unwinding, no `finally` blocks, no buffer flush. The resume phase is a brand-new interpreter that knows only what is on disk.

An in-process `try/except` would test a polite exception, not a crash. The ledger is `fsync`ed on every append precisely because the process may die between one line and the next.

```
crash_simulated  global_step=16 exit_code=137 last_checkpoint_id=ckpt-main-000012
subprocess_exit  exit_code=137 expected=137
[PASS] crash_produced_nonzero_exit  exit_code=137 mechanism=os._exit, no unwinding or flush
```

### 2.2 Rollback appends; it never deletes

When the resumed process recovers from `ckpt-main-000012`, the batches the dead process consumed at steps 12–15 never reached the weights, so they must be re-consumed. The tempting implementation is to truncate the ledger back to the checkpoint's offset. That destroys the record of what the crashed process actually did — exactly the evidence an incident review needs.

Instead a `rollback` event is **appended**, naming the offset range it orphans:

```
ledger_rolled_back  to_offset=86 from_offset=135 orphaned_events=49 orphaned_steps=[12,13,14,15]
```

`effective_events()` reconstructs the logical stream by honouring those markers; the file on disk keeps the full history. This is the transaction-log pattern the session's references point at (LakeFS / Iceberg / Delta): the log is the truth, and a rollback is another entry in it.

Every ledger is additionally **hash-chained** — each event stores `H(prev_hash ‖ event)`. Editing, reordering or deleting any historical event invalidates every hash after it, and `verify_chain` reports the exact index where the break occurs. Without this, "append-only" is a naming convention enforced by nothing. Three tests corrupt a committed ledger and assert the break is found.

### 2.3 Resume correctness is defined against model state

"No skipped or repeated batches" has to be measured against *what the weights saw*, not against wall-clock order. Steps 12–15 are re-run after the resume, and that is **correct**, not a repeat: the checkpoint predates them.

The proof compares the **full batch content hash** — every token, mask bit, segment id and position id in every microbatch — that the crashed process recorded at each step against what the resumed process produced:

```
[PASS] resume_next_batch_matched            expected_next_step=12 actual_next_step=12
[PASS] resume_no_skipped_or_repeated_batches  skipped=[] repeated=[] orphaned=[12,13,14,15]
[PASS] resume_batch_hashes_identical        steps_compared=4 mismatches=0
```

Comparing batch **ids** would be far weaker — ids are derived from the step number and would match even if the data behind them differed. Comparing content hashes is what catches a lost carry buffer or a reshuffled lane order.

### 2.4 Replay goes back to the shards

Replay does not diff two in-memory objects, and it does not re-run the packer on the same inputs. It takes the ledger event, reads the recorded coordinates — shard id, offset within the shard, length, offset within the window, segment id — re-opens the **immutable shard files**, re-cuts the spans, rebuilds the arrays, and compares the content hash to the original.

Re-running the packer would prove the packer is deterministic. Rebuilding from coordinates proves the stronger thing: *the tokens that trained the model at that step are still there, unchanged, and can be produced again*. It also works for the concatenating policies, where a document is split across two windows and re-packing from whole documents would reproduce neither.

All three things the assignment asks a replay to prove are checked separately: the **batch id** is recomputed from `branch/step/rank/microbatch` rather than read back, the **token spans** are rebuilt from coordinates and compared string-for-string, and the **hashes** cover both the loss mask and the full window content.

```
[PASS] replay_hash_matched  interval=[6,12] microbatches_replayed=24
       batch_ids_verified=24 token_spans_verified=72 mismatches=0
```

A test mutates one byte of a consumed shard and asserts the replay **fails** — a check that cannot fail is not a check.

### 2.5 Evidence is re-derived, never reported

`src/erav6/evidence.py` runs last, **in its own process**, and never imports the trainer. Its only inputs are files: `run.log`, `manifests/`, `ledgers/`, `checkpoints/`, `performance.json`, the frozen tokenizer, and the shards. Each of the eleven requirements is recomputed from those files:

- the tokenizer hash is recomputed from `tokenizer.json` and compared against what every manifest recorded;
- the firewall result comes from walking every consume event and intersecting shard ids with the never-train registry;
- packing correctness is checked by **rebuilding all 192 windows from ledger coordinates** and re-testing five mask invariants on the rebuilt arrays;
- every throughput rate is re-divided from the raw counters in the same file.

If a claim cannot be reconstructed from the bundle, the row reports FAIL. That is the intended behaviour: an unverifiable claim is not evidence.

This was verified adversarially. Corrupting a copy of the bundle in four different ways — altering a recorded batch hash, altering a loss-mask hash, multiplying a throughput rate by ten, and injecting a never-train shard id into a consume event — makes exactly the corresponding requirement flip to FAIL, while an untouched copy still reports 11/11. A generator reporting stored booleans would have stayed green through all four.

---

## 3. Architecture

```
corpus/documents.jsonl        78 documents, 11 lanes, role-tagged segments
        │
        ▼  tokenizer.py (frozen, hashed: b026dbab…)
shards.py         immutable .bin (uint32 tokens) + .idx.json (role spans)
        │
        ├──▶ registry.py     validation/test docs → 266 contamination fingerprints
        ▼
manifest.py       15-field manifest + 9-check admission gate  →  25 admitted / 4 blocked
        │
        ▼
mixture.py        Session-5 stages → per-step integer lane quotas + floor + ramps
        │
        ▼
dataloader.py     (branch, seed, step) → batch          ── pure function ──┐
   │  packing.py    6 policies, masks, position ids                        │
   │  opus.py       scored from the live model                             │
   ▼                                                                       │
trainer.py        forward/backward, honours segment mask                   │
   │                                                                       │
   ├──▶ ledger.py     consumption · opus · learning · firewall · token_trace
   ├──▶ learning.py   probe before/after every step, per-token perplexity
   └──▶ checkpoint.py weights + optimizer + RNG + loader state + LEDGER OFFSET
                                                                           │
audit.py          firewall sweep · interval queries · replay ◀─────────────┘
evidence.py       re-derives all 11 requirements from the artefacts alone
```

### The batch is a pure function

```
batch at (branch, seed, step) = f(plan, admitted shard pool, checkpointed loader state)
```

Everything about crash recovery follows from that. The loader state that must survive a crash is larger than it first looks — and the classic resume bug is forgetting the last item:

| State | Why losing it breaks resume |
|---|---|
| `next_step` | obvious |
| `lane_cursor` | position in each lane's shuffled order |
| `lane_epoch` | also *selects* the shuffle — wrong epoch changes the **order**, not just the position |
| `carry` | samples drawn from a shard but not yet packed into an emitted window |
| OPUS `deferred` + `recent` | the deferred queue and the redundancy window |

Dropping `carry` on resume silently **skips** data; re-drawing it silently **repeats** it. Everything still looks right at the step boundary, which is why the proof compares content hashes rather than batch ids.

---

## 4. Shards, manifests and the admission gate

A shard is a sealed object: a flat little-endian `uint32` token array plus a side index recording which document each region came from and **what role every token plays**. Every token belongs to exactly one contiguous span, and each span carries its loss flag — that is where "tool observations are context-only" stops being a design note and becomes a bit in a file. A test asserts the spans tile every document exactly, because a gap would be a token with no defined loss policy.

Shards are grouped by **(lane, split, licence)** and never straddle any of the three. Lane, because the scheduler selects at lane granularity. Split, because permissions differ. Licence, because a shard's tier is the *worst* licence among its members — mixing one unclear-licence document into an otherwise permissive shard takes the whole shard down with it.

**Content hash covers the bytes AND the index.** Same tokens with a shifted span index is a different training object, because the spans decide which tokens carry loss. Both halves are hashed; a test flips one span's loss bit and asserts the identity changes.

The gate runs **nine checks**, deciding on metadata and bytes — never on how the text reads:

| Check | Blocks |
|---|---|
| `manifest_complete` | missing required fields |
| `tokenizer_hash_verified` | token ids whose tokenizer is unknown or different |
| `content_hash_verified` | a shard edited since it was written |
| `license_permitted` | `unknown` / `noncommercial` / `restricted` |
| `cleaning_lineage_known` | unrecognised cleaning pipeline |
| `dedup_passed` | not through exact+fuzzy dedup |
| `not_never_train` | registered benchmark data |
| `is_training_split` | validation data (readable for eval, never gradient-bearing) |
| `eval_overlap_clean` | a training doc sharing n-grams with a benchmark item |
| `no_canary_strings` | a canary in a training shard |

Result on this corpus: **25 admitted, 4 blocked**, each for a *distinct, single* cause where it matters — `shard-english-train-005` for contamination alone, `shard-english-train-006` for licence alone.

**Immutability is demonstrated, not asserted.** The demo copies an admitted shard, flips one byte of one token, and re-verifies:

```
[PASS] shard_immutability_enforced  original_hash=791930056d5dcfb2  tampered_hash=187dcb82d40b611a
```

"Immutable" does not mean editing is impossible. It means editing is **detected**.

---

## 5. The evaluation firewall

Three permission levels in one registry:

| Split | Permission |
|---|---|
| `train` | admitted into the gradient-bearing stream |
| `validation` | **readable** for evaluation, never gradient-bearing |
| `test` | `never_train`; no read path inside training at all |

The validation/test distinction is the one usually implemented sloppily. Validation data is *allowed* into a forward pass, so the protection it needs is not "never load this" but "never let this reach an optimizer step". That is enforced structurally — `open_for_eval` stamps every read with a purpose and refuses any gradient-bearing one:

```
[PASS] validation_read_allowed_gradient_denied  eval_read=allowed gradient_read=denied
```

**Contamination is detected, not declared.** The corpus contains `eng-contam-01`, an ordinary-looking English document that quotes a benchmark item, with its declared status left as `unscanned` on purpose — the scanner has to make the call itself. Fingerprints are **24-token n-grams**, because tokens are what actually enter a batch. The scanner found **47 shared n-grams** with `test-bench-002` and blocked the shard.

The final audit does not check intent. It walks every consume event on both branches, collects every shard id that actually entered a gradient-bearing batch, and intersects with the never-train set:

```
[PASS] no_eval_data_in_training  never_train_consumed=[] blocked_consumed=[] distinct_shards_consumed=24
```

---

## 6. Packing, masks and position ids

Every window carries five aligned arrays plus provenance. The subtle one is `labels`: loss at position *i* is a prediction of the token at *i+1*, so "is this position loss-bearing" really means "is the **next** token loss-bearing, and is it legitimately predictable from here". Three things make it illegitimate, all handled in one place for every policy:

1. the next slot is padding,
2. the next token starts a different **isolated** sample,
3. the next token is **context-only** (a tool observation or an SFT prompt).

All six policies are implemented and measured on real lane data:

| Policy | Isolation | Position ids | Used for |
|---|---|---|---|
| `pad_only` | per sample | restart | maximum safety, minimum efficiency |
| `concat_and_chop` | none — one segment | contiguous | plain prose, Indic, forums |
| `greedy` | per sample | restart | math |
| `best_fit` | per sample | restart | code, verified Indic |
| `structure_preserving` | per sample | restart | agentic, reasoning |
| `long_context` | per sample, long samples alone | restart | any lane with an oversized sample |

`loss_across_boundary` is reported on every window and **must be zero for every isolating policy**. The evidence generator rebuilds all 192 consumed windows and re-checks it, plus label shift, no-loss-on-padding, per-segment position restart, and loss-mask hash equality. **0 violations.**

The model actually uses this. `build_attention_bias` constructs the mask from `segment_ids` — `allowed(i,j) ⟺ j ≤ i ∧ seg[i] = seg[j] ∧ seg[i] ≠ 0` — so co-packed samples are genuinely invisible to each other rather than nominally isolated. A system that emits beautiful metadata and then feeds a plain causal mask to the model has demonstrated nothing.

### Loss masking, measured

The Session-5 §2.1 claim, confirmed by the P5 proxy, enforced here at the bit level on a real agentic window:

| Role span | Declared loss | Mask sum |
|---|---|---|
| `user` (the request) | 0 | **0 / 40** |
| `think` | 1 | 49 / 50 |
| `tool_call` | 1 | 32 / 33 |
| **`tool_obs`** (tool output) | **0** | **0 / 36** |
| `tool_call` | 1 | 21 / 22 |
| **`tool_obs`** | **0** | **0 / 31** |

The off-by-one on the loss-bearing spans is correct: the last token of a `think` span predicts the next role marker, which is structural and not loss-bearing.

A document cannot smuggle in a control token — role markers are emitted structurally from declared roles, never by matching strings in the text. A test feeds a `tool_obs` span containing the literal text `<|assistant|>` and asserts the mask stays zero.

### A bug worth recording

The first implementation cut oversized structured samples at the last role boundary that fit. For a reasoning trace — `<user><think …400 tokens…><assistant>` — the role boundaries sit at offsets 0, 1, 2, 40, 41 and 350, so the largest boundary under a 256-token window is **41: the end of the prompt**. Every reasoning window came out with 1904 truncated tokens and a loss density of exactly **zero**.

The fix is a floor: a boundary cut is only taken if it retains ≥60% of the window, otherwise the sample is cut hard, keeping a damaged tail but keeping the gradient. A regression test asserts no policy can emit a window with no loss-bearing tokens. At Session-5 scale this case largely disappears, because these lanes get 32K–256K windows precisely so a trace has room to finish.

---

## 7. Mixture, floors and scarcity

The compiler turns shares into an integer number of sequences per lane per step.

**Cumulative-deficit allocation, not per-step largest remainder.** Per-step largest remainder is the obvious approach and it is wrong at small batch sizes. With 8 sequences per step, a lane whose target share is 4% asks for 0.32 sequences, loses the remainder contest to bigger lanes every single step, and is allocated **zero for the entire run**. Measured before the fix:

| Lane | Target | Realised (per-step LR) | Realised (deficit) |
|---|---|---|---|
| `indic_synthetic` | 0.040 | **0.000** | 0.036 |
| `forums` | 0.030 | 0.010 | 0.026 |
| `multilingual` | 0.030 | 0.010 | 0.026 |
| `english` | 0.300 | 0.318 | 0.292 |

Carrying the fractional debt fixes it: a 4%-share lane accumulates 0.32 per step and is served on roughly every third step — delivered as a share of the **run**, not forced into every batch. Max drift against the Session-5 target fell from 0.055 to **0.012**.

**The protected floor is enforced twice.** The quota compiler reserves floor slots by promotion — taking a sequence from whichever lane is furthest *above* its own target and giving it to the native-Indic lane furthest *below* — and every promotion is recorded. Then, after OPUS selects, a second pass rescues the floor if the selector still under-delivered. Result: **0 violations across all 24 steps, minimum delivered share 12.5%**.

At 8 sequences per batch, 8% of a batch is 0.64 sequences, which cannot be delivered. Rounding up to 1 gives 12.5% — above the floor, as a floor requires. At Session-5's 512-sequence batch the floor is 41 sequences and this granularity artefact disappears.

**Every deviation from plan must be attributable.** Three lanes — `english`, `indic_translated`, `indic_synthetic`, exactly the lanes Session 5 caps at 1.0 epoch as having the least slack against supply — exceeded their scaled epoch cap and invoked their declared fallback. That means realised shares depart from the compiled quota, and that is the plan *working*, not failing. So the pass criterion is not raw drift but **unexplained** drift: the part no logged event accounts for.

```
[PASS] mixture_deviation_fully_attributed
       max_unexplained_drift=0  tolerance=0.02
       max_raw_drift=0.114583  substitution_events=16  sequences_substituted=26
```

`english` came in at 0.177 against a planned 0.292 — because it hit its epoch cap and its declared `reduce_share` response moved 26 sequences to lanes with headroom, in 16 logged events. **Unexplained drift is exactly 0.**

**The anneal reserve is real.** One shard per Tier-A lane is invisible to the loader until stage S4. The audit checks it directly — zero consumption events before S4, non-zero within it:

```
[PASS] anneal_reserve_respected
       reserved=[shard-agentic-train-002, shard-indic_verified-train-001, shard-reasoning-train-002]
       leaked=[]
```

---

## 8. OPUS and its audit trail

Every candidate is scored by running the **current model** over it. That is the part that cannot be faked: utility is a measured property of the model at this step, so the same candidate scores differently early and late, and a grader reading the ledger sees the scores move. A test asserts the scores actually vary — a selector whose scores never change is not measuring anything.

```
utility = z(mean loss under the live model)          # learning headroom
        − 0.35 · redundancy(recently consumed shards)
        + 0.25 · stage_fit(lane, current stage)
```

**740 decisions** recorded across both branches, every one with a status, a score, a reason, its shard ids and span ids, the checkpoint it was scored against and the proxy version:

| Outcome | Count | Meaning |
|---|---|---|
| accepted | 256 | entered the gradient-bearing stream |
| rejected | 273 | scored, not used, **reason retained** |
| deferred | 211 | valuable but not wanted now — re-offered later |
| floor_override | *see below* | rescued by the protected floor |

Rejection reasons: `duplication` 103, `stage_mismatch` 101, `quota_pressure` 86, `defer_expired` 133, `low_proxy_utility` 2.

**Deferred candidates provably come back.** The queue stores span *coordinates*, not tensors, so a re-offered candidate is rebuilt from the immutable shards by the same path replay uses — and is therefore byte-identical to the one that was deferred. On `main`: **185 deferral decisions, and 162 records of a deferred candidate being re-offered** (a candidate can be deferred more than once, which is why the second number is not bounded by the first). Deferrals that age out past `defer_max_age` are recorded as `defer_expired` rather than silently dropped — 133 of them.

### The floor override, and why it needed a drill

The main run records **zero** floor overrides, and the reason is worth stating plainly rather than engineering around.

This system's proxy is loss headroom. With a byte-level tokenizer, Devanagari and Tamil are the **highest-loss content in the corpus**:

| Lane | Median utility |
|---|---|
| `indic_verified` | **+1.218** |
| `indic_unverified` | **+1.203** |
| *global median* | −0.538 |
| `english` | −0.859 |
| `code` | −0.852 |

The selector *prefers* native Indic. It never needs rescuing, so the override never fires. That is a genuine result — but it leaves the floor untested, and Session 5's argument for the floor is about a **different** proxy: *"English-tuned quality proxies systematically reject Indic script, code-mixing and local content, so an unprotected selector starves the lane."*

So `floor_drill.py` builds exactly that proxy and runs it as a labelled counterfactual — Session 5's P2 experiment in miniature. Both arms score the **same candidate pool** with the **same model** at the **same step**; the only difference is the floor:

| Arm | Native-Indic sequences | Floor met | Lanes selected |
|---|---|---|---|
| **A — unprotected** | **0** | ✗ | english 6, math 2 |
| **B — protected** | **1** | ✓ | code 1, english 3, forums 1, indic_unverified 1, math 1, multilingual 1 |

```
[PASS] protected_floor_override_demonstrated
       unprotected_native_indic=0  protected_native_indic=1  floor_required=1  n_overrides=1
```

The bias magnitude was set deliberately. At −3.0 per Indic lane, native Indic still scored *above* the pool average (−0.39 against −0.58) and the unprotected arm picked it up anyway. A proxy that "systematically rejects" a lane has to outweigh the evidence, not tie with it, so the penalty is set larger than the full observed spread of the headroom score. Drill decisions carry the proxy version `opus-proxy-1.0.0-lane-biased-drill` so they can never be mistaken for real ones.

---

## 9. Crash, resume, replay, fork

The demonstration, in order:

| Step | What happens | Evidence |
|---|---|---|
| 0–11 | branch `main` trains; checkpoints at 6 and 12 | `[PASS] checkpoint_saved` ×4 |
| 12–15 | four more steps consumed and recorded | ledger |
| 16 | **`os._exit(137)`** mid-step | `exit_code=137` |
| — | fresh process starts, finds `ckpt-main-000012` | |
| — | ledger rolled back to offset 86, orphaning steps 12–15 | `rollback` event |
| — | **binding verified**: ledger head at offset 86 == what the checkpoint recorded | `[PASS] checkpoint_ledger_binding_verified` |
| 12–23 | resumed run completes | 4 overlapping hashes identical |
| — | interval [6,12) replayed from shards | 24 microbatches, 72 spans, 0 mismatches |
| 6→9 | fork onto a new branch with a new data seed | 4 shared steps, **4 diverging** |

**Checkpoints bind model state to data state.** Every checkpoint carries the ledger offset, the ledger head hash at that offset, the full loader state, the data seed, the tokenizer hash and the admitted-pool hash. `restore` re-checks the head hash against the live ledger and **raises** rather than resuming into a stream the weights were not trained on. A test mutates a checkpoint's recorded offset by one and asserts `LedgerBindingError`.

**A fork must actually fork.** The branch name is part of every shuffle seed, so a fork diverges by construction while remaining reproducible on its own terms. The check requires the divergence point to be recorded *and* the streams to genuinely differ — a fork that reproduced the parent's stream would be a replay wearing a fork's name. All 4 shared steps differ.

**Audit queries** answer the Session-6 questions from the ledgers alone: which shards influenced the model over a step or token range (21 shards, 39 documents in [6,12)), which batch preceded the largest loss spike, whether every hash chain verifies, and whether any never-train shard ever reached a gradient.

---

## 10. The two-way learning ledger

The consumption ledger records what the model saw. The learning ledger attaches the outcome back to the data.

**Loss deltas are measured, not inferred.** A fixed probe window is cut from each shard, the model is evaluated on it immediately **before** the optimizer step and again immediately **after**, and the difference is recorded against that shard. Same probe, same mask, two forward passes either side of one update — **212 exposures, all 212 with a measured probe delta.**

The probe is deliberately a *held-back* slice of the shard rather than the batch just consumed. Measuring on the consumed batch would measure memorisation of those exact tokens; measuring on a different slice of the same shard measures whether anything transferred. Shards are then classified `useful` / `neutral` / `harmful` from those deltas — all three classes appear in this run.

**Token-level traces** record, for every loss-bearing token in sampled batches: token id, decoded preview, position, document id, shard id, lane, special/boundary/mask flags, cross-entropy, perplexity, curriculum stage, model phase, tokens-consumed-when-seen, the checkpoints either side, the OPUS score and the repeated-pass number. **336 records**, every one linked back to a source document. A test recomputes `exp(cross_entropy)` and checks it matches the recorded perplexity.

The ledger also runs the proxy-bias check Session 6 asks for: shards OPUS scored below zero that the measured probe deltas say were useful — i.e. places the selector may be undervaluing a capability.

---

## 11. Throughput

Every number in `performance.json` is a real counter or a real wall-clock timer, and every headline rate is reconstructible from the raw counts in the same file. The evidence generator re-divides them and cross-checks packing utilization against the token counts independently recorded in the consumption ledger.

| Metric | Value |
|---|---|
| Token positions computed | 57,344 |
| Real tokens | 42,696 |
| **Loss-bearing tokens** | **40,139** |
| Raw tokens/s | 2,114.0 |
| Real tokens/s | 1,574.0 |
| **Useful loss-bearing tokens/s** | **1,479.7** |
| Packing utilization | 0.745 |
| Padding waste | 0.255 |
| Context-only fraction | 0.060 |
| OPUS acceptance rate | 0.346 |
| Shard cache hit rate | 1.00 |

The counts are deterministic; the per-second rates are wall-clock and will differ on your machine. The ratios and the counts are what the evidence checks, and both are reconstructible from the raw counters in the same file.

The four token rates and the gaps between them are the point: a system optimised for `raw_tokens_per_s` can be losing badly on `useful_loss_bearing_tokens_per_s`. Here 25.5% of every computed position is padding and a further 6.0% is context-only, so **70%** of the compute carries a gradient.

**Token counts come from the ledger, not from the workers.** The crashed process died via `os._exit` and never wrote a summary, so summing the worker summaries would silently omit the twelve steps it completed before the crash — an undercount of a third of the run. The ledger was fsynced on every append, so it has them.

**The selector is 90% of data-loading time**: 0.736s of the 0.816s spent building batches. That is the honest cost of putting OPUS inside the data path — it scores 740 candidates to accept 256, and the discarded forward passes are real compute.

Measured packing comparison (English lane, 256-token window): `concat_and_chop` reaches 1.000 utilization against `pad_only`'s 0.800 — and pays for it with 7 boundary crossings per 6 windows, which is exactly the trade the isolating policies refuse to make for structured data.

---

## 12. Evidence and tests

```
submission_artifacts/
├── run.log            full event sequence + 31 [PASS] markers, across all subprocesses
├── evidence.json      11 requirements, each with how_checked, evidence paths, numbers
├── evidence.md        human-readable summary table
├── performance.json   counters, timers, four token rates, per-policy comparison
├── manifests/         29 shard manifests + _index, _eval_registry,
│                      _mixture_timeline, _packing_report
├── ledgers/           consumption · opus · learning · firewall · token_trace,
│                      per branch, hash-chained, + _audit_report, _floor_drill
└── checkpoints/       binding records (JSON) + weights (.pt)
```

| Requirement | Result | Key numbers |
|---|---|---|
| Tokenizer integrity | **PASS** | 29 manifests checked, 0 mismatched |
| Shards, manifests, immutability | **PASS** | 29 shards, 25 admitted, 4 blocked, 0 hash failures |
| Evaluation firewall | **PASS** | 4 blocked, 24 consumed, 0 never-train reached a gradient |
| Packing correctness | **PASS** | 192 windows rebuilt, **0 violations** |
| Mixture, floors, curriculum | **PASS** | unexplained drift **0**, floor violations **0** |
| OPUS audit trail | **PASS** | 740 decisions, 162 deferred returned, 1 floor override in drill |
| Learning trace | **PASS** | 212 exposures with probe deltas, 336 token records |
| Crash recovery | **PASS** | exit 137, expected step 12 = actual 12, **0 hash mismatches** |
| Replay and fork | **PASS** | 24 batch ids, 72 spans, all hashes, 0 mismatches; fork diverges 4/4 |
| Throughput | **PASS** | all rates reconstructible from raw counts |
| End-to-end | **PASS** | 24 contiguous steps, 10 chains verify, 0 FAIL markers |

**97 tests**, `unittest.TestCase` so they run with zero install:

```bash
python -m unittest discover -s tests -v     # stdlib
python -m pytest tests -q                   # also works
```

`test_invariants.py` (50) builds its own fixtures and needs no prior run — tokenizer determinism and hash sensitivity, span tiling, tool-observation masking, control-token smuggling, all six packing policies' mask/position/isolation invariants, rebuild-from-coordinates round trip, quota allocation, ledger chain corruption, admission gate rules, firewall permissions.

`test_artifacts.py` (47) is an **independent second reading** of the generated bundle — it re-derives the same conclusions without going through `evidence.py`, and asserts they agree. If the generator and the tests disagree, one of them is wrong, and that is worth knowing.

---

## 13. What this does not prove

Stating the limits, because a system that claims more than it demonstrates is the failure mode this whole assignment is about.

- **The tokenizer is a proxy.** 512 byte-level entries standing in for Session 3's 196,608. What is *not* a proxy is the contract around it — frozen at build time, content-hashed, bound into every manifest, and a gate that refuses a shard whose tokenizer hash disagrees.
- **The cleaning pipeline is a stub.** `erav4-clean@4.2.1` is a hash, not a pipeline. The Session-4 admission contract is what is implemented; the cleaning itself is upstream and out of scope here.
- **The corpus is authored for this demo.** 78 documents, 14,375 tokens. It is structurally varied on purpose — role-tagged trajectories, real Devanagari and Tamil, Python source — because packing and masking are graded per data type. It is not representative of anything.
- **Epoch caps are scaled.** Session 5's caps are stated against 4T tokens of real supply; against a 14K-token corpus every lane would be in breach on step one. The caps are scaled by a single factor that preserves their *relative* tightness, and both the scaled and unscaled values appear in the compiled timeline.
- **The floor override is demonstrated by a drill, not by the main run** — for the measured reason given in §8, stated rather than engineered around.
- **`approx_idle_fraction` is an approximation.** This is a single-process demo, not an instrumented cluster; it is time not spent in a forward/backward pass, measured against the training phases only.
- **Loss goes 6.27 → 5.73 over 24 steps.** That is a real optimizer moving a real model, and it is not a capability claim. Nothing here is evidence about what the model learned, only about what the *data system* did.

---

## 14. Repository layout

```
assignment6/
├── README.md               this document
├── run_demo.py             THE COMMAND — 13 phases, subprocesses, no arguments
├── corpus/
│   ├── documents.jsonl     78 documents, role-tagged, with provenance
│   └── sources.json        13 sources with licence and provenance tags
├── tokenizer/tokenizer.json  frozen, hashed b026dbab…
├── tools/
│   ├── build_corpus.py     authored once; output is committed
│   └── build_tokenizer.py  trained once on the TRAIN split only
├── src/erav6/
│   ├── plan.py             imports the Session-5 plan; refuses to run without it
│   ├── config.py           every knob, and why each is set where it is
│   ├── hashing.py          one canonical hashing path for the whole system
│   ├── tokenizer.py        byte-level BPE, frozen and content-hashed
│   ├── shards.py           documents → immutable token arrays + role spans
│   ├── manifest.py         15-field manifest + 9-check admission gate
│   ├── registry.py         the evaluation firewall
│   ├── catalog.py          corpus in, admitted shard pool out
│   ├── mixture.py          stages → per-step quotas, floor, deficit allocation
│   ├── packing.py          six policies, masks, position ids, replay rebuild
│   ├── opus.py             selector scored from the live model
│   ├── dataloader.py       the pure function, and the state resume needs
│   ├── model.py            tiny transformer that honours the segment mask
│   ├── trainer.py          the loop; crash injection
│   ├── learning.py         probes, per-token trace, shard report cards
│   ├── checkpoint.py       model state and data state, together or not at all
│   ├── ledger.py           append-only, hash-chained, rollback by appending
│   ├── audit.py            firewall sweep, interval queries, replay, resume
│   ├── floor_drill.py      the controlled counterfactual
│   ├── perf.py             measured throughput
│   ├── evidence.py         re-derives all 11 requirements from artefacts
│   ├── runlog.py           run.log and the [PASS]/[FAIL] markers
│   └── train_worker.py     one training process; can die on command
├── tests/
│   ├── test_invariants.py  50 unit tests, no prior run needed
│   └── test_artifacts.py   45 end-to-end tests over the generated bundle
└── submission_artifacts/   generated by run_demo.py
```

### Regenerating everything

```bash
python run_demo.py                       # the whole demonstration
python run_demo.py --device cpu          # force CPU (what CI runs)
python -m unittest discover -s tests     # 97 tests
```

The corpus and tokenizer are frozen inputs, rebuilt only if you change them:

```bash
python tools/build_corpus.py && python tools/build_tokenizer.py
```

Checkpoint weight blobs (`checkpoints/*.pt`, ~10MB each) are gitignored; the binding **records** (`checkpoints/*.json`) — which carry the ledger offset, head hash, loader state and record hash that the evidence checks — are committed. `run_demo.py` regenerates the weights.

**Prior work:** [Session 3 — data & tokenizer](https://tourmaline-longma-0a9e58.netlify.app/) · [Session 4 — cleaning pipeline](https://merry-banoffee-08ed4e.netlify.app/) · [Session 5 — mixture & curriculum](../assignment5/README.md)
