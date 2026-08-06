# P5 — Loss-masking on tool observations — executed

A real, executed test of the mechanism behind README §2.1's headline claim: masking loss on
tool-output tokens ("tool observations are context-only") prevents the model from learning to
fabricate them, at no cost to actually using a tool's output correctly.

Script: [`run_p5.py`](run_p5.py). Raw output: [`results.json`](results.json).

## Setup

A synthetic tool-use trajectory, single-digit addition, fixed 11-character format:

```
Q{a}{b}  C{a}{b}  O{sum:02d}  A{parity}
prompt   tool     tool        final answer
         call     output      (model-generated, from tool output)
```

- `Q{a}{b}`: the task — add two single digits (0–9).
- `C{a}{b}`: the model's tool call, restating the arguments.
- `O{sum:02d}`: the tool's response — deterministic, environment-provided, exactly analogous
  to a real tool's return value. **This is the span under test.**
- `A{parity}`: the model's final answer — even/odd of the sum — which requires actually
  reading the tool's output correctly, standing in for using a tool result rather than just
  restating it.

100 possible `(a, b)` combinations exist; 80 are used for training, 20 are held out and never
seen during training, so generalization (not memorization) is what gets measured.

- Model: 152,832-param causal transformer (3 layers, d_model 64, 4 heads), character-level,
  vocab 16, seq_len 11.
- Two arms, matching `proxy.py`'s P5 exactly:
  - **A — masked**: loss zeroed on the tool-output span (`O` + 2 digits). The model never
    receives gradient reward for producing that content, only for reading it as context.
  - **B — full loss**: loss computed on the entire sequence, including the tool-output span —
    the naive approach the plan argues against.
- 4,000 training steps per arm, 3 seeds each (6 runs total), batch size 64.

## The two things measured

1. **Fabrication rate**: withhold the true tool output, let the model generate freely past the
   tool-call span, and check how often it produces the *correct* tool response anyway — does
   it behave as if it can compute the tool's job itself, rather than needing the tool?
2. **Task success**: *supply* the true tool output and check whether the model reads it
   correctly to produce the right final answer — does masking cost anything on the task the
   tool exists for?

Both are measured on training combos (seen) and held-out combos (never seen), across all 3 seeds.

## Results

| Arm | Fabrication (train) | Fabrication (holdout) | Task success (train) | Task success (holdout) |
|---|---:|---:|---:|---:|
| A — masked | **0.00** (0/3 seeds &gt; 0) | **0.00** (0/3 seeds &gt; 0) | 1.00 | 1.00 |
| B — full loss | 1.00 | **0.733** (0.70, 0.75, 0.75) | 1.00 | 1.00 |

Every seed agrees: Arm A fabricates the tool's output **zero times** across all 3 seeds and
both combo sets. Arm B reproduces it perfectly on training combos (it was directly rewarded
to) and **73.3% of the time on combos it never saw during training** — it has generalized the
addition function well enough to confidently "hallucinate" the tool's answer for unseen inputs
too. Task success is perfect for both arms on both combo sets — masking costs nothing.

Wall-clock: 284.9 seconds for all 6 runs (2 arms × 3 seeds) on the RTX 3050.

## Applying the plan's own decision rule

> "A must cut fabricated-observation rate by ≥50% at no BFCL cost."

Observed: fabrication rate on held-out combos drops from 73.3% (B) to 0.0% (A) — a **100%
reduction**, well clear of the plan's own 50% bar — while task success is identical (1.00 vs
1.00) in every condition. **The hypothesis is confirmed, decisively, and by all 3 seeds
independently**, not a single noisy run.

## What this does and doesn't show

- **Robust:** training with loss on a tool's output teaches a model to reproduce that output
  from memory/pattern rather than needing the tool — confirmed directly, not asserted. Masking
  that loss eliminates the effect entirely in this setup, with zero measurable cost to using a
  genuinely supplied tool output for the actual task.
- **What it does not show:** this is single-digit addition inside an 11-character synthetic
  trajectory, not real agentic tool use (terminal commands, API calls, multi-step trajectories)
  and not the real BFCL / τ²-bench metrics the plan's full P5 design calls for. The *mechanism*
  — masking prevents the model from internalizing a shortcut around the tool — is confirmed
  cleanly and unambiguously; the *magnitude* on real agent trajectories (real tool complexity,
  real vocabulary, real task diversity) still needs the 1B/20B-token run specified in §10. What
  changes: this is no longer an assumption behind the §2.1 "0.084T supervised" calculation —
  it is a mechanism that has been shown, on a real executed run, to work exactly as claimed.
