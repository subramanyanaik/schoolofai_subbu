# P4 v2 — Mixture-transition gradient stability — deepened

v1 ([results.md](results.md)) established the mechanism on synthetic Markov-chain "lanes"
with a single seed and three ramp lengths. v2 strengthens the same test on every axis:

| | v1 | v2 |
|---|---|---|
| Data | synthetic Markov chains | **real text** — this repo's own prose vs. its own Python source |
| Model | 817,920 params | **2,733,696 params** (~3.3× larger) |
| Ramp lengths tested | 3 (0, 60, 300) | **8** (0, 20, 40, 60, 100, 150, 200, 300) |
| Seeds per point | 1 | **3**, reported as mean ± population std |
| Question answered | "does ramping help, roughly?" | **"what is the shortest ramp that suppresses the shock?"** |

Script: [`run_p4_v2.py`](run_p4_v2.py). Raw output: [`results_v2.json`](results_v2.json).

## Setup

- **Lane A (prose)**: this repo's own `README.md` + `SPECIFICATION.md` + `SETUP.md` —
  64,120 real English characters.
- **Lane B (code)**: this repo's own `config.py`, `budget.py`, `mixture.py`, `curriculum.py`,
  `floors.py`, `proxy.py`, `sensitivity.py`, `validate.py`, `export.py`, `run_plan.py` —
  66,581 real Python characters.
- This is a genuine, naturally-occurring analogue of the plan's actual S1 (English-heavy) →
  S2 (code-heavy) curriculum transition (§4), not an abstraction standing in for it — real
  natural-language statistics on one side, real source-code statistics on the other.
- Model: 2.73M-param causal transformer (6 layers, d_model 192, 6 heads), character-level,
  vocab 119 (the union of characters actually appearing in both corpora), seq_len 96.
- Each run: 800 training steps, batch size 48. The model trains on prose alone for the first
  500 steps (loss falls from 4.94, the near-uniform-guess floor at ln(119) ≈ 4.78, to ~2.02 —
  genuine convergence, not a still-random model), then the lane-B share is introduced at
  step 500 according to the ramp schedule under test.
- Ramp sweep: 0 (abrupt step), 20, 40, 60, 100, 150, 200, 300 steps, each straddling the
  transition symmetrically (half before, half after — same rule as the plan's own ramps, §4.2).
- 3 seeds per ramp length, 24 runs total.
- Metric: peak gradient-norm over the transition window ÷ pre-transition steady-state median
  gradient-norm (mean ± population std across seeds); count of loss values exceeding 1.5×
  the pre-transition median loss in that window.

## Results

| Ramp (steps) | Mean grad-norm multiplier | Std | Mean loss spikes |
|---:|---:|---:|---:|
| 0 (abrupt) | **6.038×** | ±0.917 | 1.7 |
| 20 | **1.692×** | ±0.085 | 0.0 |
| 40 | **1.780×** | ±0.312 | 0.0 |
| 60 | **1.741×** | ±0.051 | 0.0 |
| 100 | **1.503×** | ±0.055 | 0.0 |
| 150 | **1.457×** | ±0.017 | 0.0 |
| 200 | **1.519×** | ±0.086 | 0.0 |
| 300 | **1.472×** | ±0.027 | 0.0 |

Wall-clock: 1,562.8 seconds (26.0 minutes) for all 24 runs on the RTX 3050.

## Reading the curve honestly

Two findings, and they're different in kind:

1. **The overwhelming majority of the benefit comes from having *any* ramp at all.** Going
   from an abrupt step to just a 20-step ramp cuts the gradient shock from 6.04× to 1.69× —
   a 72% reduction — and eliminates loss-spikes entirely (1.7 → 0.0). This is the single
   biggest, most confident result in the sweep: std at ramp=0 is 0.92 and at ramp=20 is 0.09,
   so the two are cleanly separated even accounting for seed noise.
2. **Beyond ~20 steps, the curve is flat and noisy, not a clean monotonic decline.** Values
   from ramp=20 through ramp=300 sit in a 1.45×–1.78× band with no consistent trend — ramp=40
   (1.78×) is noisier and nominally *worse* than ramp=20 (1.69×) and ramp=60 (1.74×), and
   ramp=200 (1.52×) bumps back above ramp=100 (1.50×) and ramp=150 (1.46×) before ramp=300
   drops again. Per-point std in this region (0.02–0.31) is often comparable to the gaps
   between neighboring points, so this is seed noise, not a real reversal.

**Applying the plan's pre-declared 1.5× spike threshold mechanically**, the first ramp length
whose mean drops below it is **150 steps**. But given point 2, that exact number should be
read as "roughly 100–150 steps," not as a precise cutoff — the honest statistical claim the
data supports is narrower and stronger than a single threshold: **nearly all of the
protection ramping buys is captured by a short ramp, and lengthening it well beyond that
point buys comparatively little, within the noise of this setup.**

## What this changes vs. v1's conclusion

v1 (synthetic, single-seed) found the abrupt step spiked 2.84× and reasoned the plan's
100B-token band was likely over-specified relative to the 20B-token alternative. v2, with
real text, a larger model, and repeated seeds, finds a **larger absolute shock** (6.04× vs
2.84× — real language produces a bigger, more realistic gradient shock than the synthetic
Markov chains did) and a **sharper, better-supported version of the same conclusion**: the
plan's own decision rule ("keep the 100B band only if the abrupt arm spikes ≥5× AND the
20B-analog arm also spikes") does not hold at this scale — the abrupt arm clears the 5× bar
convincingly (6.04×), but the short-ramp arm does not spike at all (0 loss-spikes at every
non-zero ramp length tested, starting at 20 steps). The evidence now points specifically at
a short ramp, not just "shorter than 100B."

## What this does and doesn't show

- **Robust:** an abrupt mixture-share transition shocks a model that has specialized on the
  prior distribution — now shown on real text, not just synthetic chains, at a larger model
  size, and with variance quantified rather than a single run's noise.
- **Not shown:** the real 100B/20B-token band width for a 15B-active-parameter MoE. This
  model is smaller by roughly four orders of magnitude and trains on real character counts,
  not real production token counts, so the sweep's step numbers cannot be read directly as a
  token-equivalent recommendation for §4.2. What changes is the confidence in the *shape* of
  the answer: the real 3B/60B-token P4 run specified in §10 should test a **shorter** band
  than 100B tokens as its primary candidate, not the 100B width as a default, because two
  independent runs (synthetic and real-text) now agree that most of the protection saturates
  early.
