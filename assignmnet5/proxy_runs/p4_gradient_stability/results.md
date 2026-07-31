# P4 — Mixture-transition gradient stability — executed

A scaled, real, executed run of the P4 mechanism, not a re-run of the full 3B/60B-per-arm
design in [`proxy.py`](../../proxy.py). Same three arms, same decision rule, run at a size
a single consumer GPU can actually train: an 818K-parameter transformer (4 layers, d_model
128, 4 heads) on two synthetic Markov-chain "lanes" standing in for two curriculum-stage
mixes, on an NVIDIA RTX 3050 Laptop GPU (4GB VRAM).

Script: [`run_p4.py`](run_p4.py). Raw output: [`results.json`](results.json),
[`traces.json`](traces.json) (full per-step gradient-norm and loss traces).

## Why synthetic lanes are a fair stand-in here

P4 tests a scheduling mechanism — does an abrupt change in a curriculum-stage's data mix
shock the gradient, and does ramping the change prevent it — not a claim about which
languages or lanes are involved. Two Markov chains over a shared 64-token vocabulary, sharpened
so each has a strong, distinct, learnable "accent," reproduce the property that actually
drives the effect (a sharp shift in the input token distribution the model has specialized
against) without needing curated real-language data to test a scheduling question.

## Setup

- Model: 817,920 params, causal transformer, seq_len 64, vocab 64.
- Each condition: 1,000 training steps, batch size 64 → ~4.1M synthetic tokens/condition
  (~12.3M tokens total across all three conditions).
- Transition point: step 500. Model is given 500 steps on lane A alone first — enough to
  converge (loss drops from 4.31, the uniform-guess floor at ln(64), to ~2.9) — so the
  switch to lane B is a genuine distribution shock to a model that has actually specialized,
  not a shock to a still-random model.
- Three arms, matching `proxy.py`'s P4 exactly:
  - **B — abrupt step**: lane share flips 0→1 in a single step.
  - **C — short ramp**: linear ramp over 60 steps, straddling the transition (proportionally
    analogous to the plan's 20B-token ramp).
  - **A — long ramp**: linear ramp over 300 steps, straddling the transition (proportionally
    analogous to the plan's 100B-token ramp — same 5:1 ratio to the short ramp as 20B:100B).
- Metric: peak gradient-norm over the transition window ÷ pre-transition steady-state median
  gradient-norm; count of loss values exceeding 1.5× the pre-transition median loss in that
  window. Same two metrics named in the original P4 spec.

## Results

| Arm | Ramp (steps) | Baseline median grad-norm | Peak grad-norm | **Multiplier** | Loss spikes |
|---|---:|---:|---:|---:|---:|
| B — abrupt step | 0 | 0.327 | 0.929 | **2.84×** | 11 |
| C — short ramp | 60 | 0.325 | 0.421 | **1.29×** | 0 |
| A — long ramp | 300 | 0.323 | 0.359 | **1.11×** | 0 |

Wall-clock: 109.5 seconds for all three conditions on the RTX 3050.

## Applying the plan's own decision rule

> "Keep the 100B band only if the abrupt arm spikes ≥5× AND the 20B arm also spikes. If 20B
> is sufficient, shorten the band and reclaim schedule."

At this scale: the abrupt arm spiked 2.84× — real, but under the 5× bar the plan set for
itself. The short-ramp arm did not spike at all by either metric (1.29× multiplier, zero
loss-spikes). Taken literally, this run's numbers point toward **shortening the band**, not
confirming the 100B width — which is exactly the outcome arm C exists to catch.

## What this does and doesn't show

- **Robust finding:** an abrupt mixture-share transition produces a measurable gradient and
  loss shock in a model that has specialized on the prior distribution, and that shock
  shrinks monotonically as the transition is ramped (2.84× → 1.29× → 1.11× as ramp length
  goes 0 → 60 → 300 steps). This confirms the *mechanism* the plan's §4.2 ramping rule
  depends on — it is not an assumption, it reproduces under a real, executed test.
- **What it does not show:** the correct ramp width in real tokens. This model is ~3,700×
  smaller than the flagship's 15B active params, trained on ~4.1M tokens per arm against the
  real proxy's 60B, on synthetic Markov chains rather than real English/code/Indic token
  distributions. A gradient-shock mechanism confirmed at this scale is real evidence the
  mechanism exists; it cannot set the "100B vs 20B tokens" number for a 15B-active MoE — that
  still needs the real 3B/60B-token P4 run specified in §10. What changes is the confidence
  that ramping *at all* is worth the schedule cost, and a concrete reason (this run's own
  numbers) to test the 20B-scale band seriously rather than assume the wider one by default.
