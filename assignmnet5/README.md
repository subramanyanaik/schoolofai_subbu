# ERA V5 — Mixture & Curriculum Specification

[![validate-plan](https://github.com/USERNAME/REPO/actions/workflows/validate.yml/badge.svg)](https://github.com/USERNAME/REPO/actions/workflows/validate.yml)
![checks](https://img.shields.io/badge/validation-49%2F49%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.12-blue)
![deps](https://img.shields.io/badge/dependencies-none-lightgrey)

**Session-5 assignment.** A defended token budget for every capability lane, the Indic split across all four provenance tiers, a protected floor, an anneal reserve, difficulty bands, and five proxy experiments — two of them executed on real GPU hardware, not just specified (§10).

**Every number below is computed by the code in this repo, not typed by hand.**

```bash
git clone <this-repo> && cd erav5-mixture/src
python3 -m erav5.run_plan        # regenerates every figure in this README
echo $?                          # 0 = all 49 validation assertions pass
```

No dependencies. Python 3 stdlib only.

| | |
|---|---|
| **Flagship** | V5-Base — 120B MoE, 15B active |
| **Train budget** | **4.00T tokens** (computed from compute, §1) |
| **Collection target** | 8.00T cleaned = 2.0× headroom for the selector |
| **Indic lane** | 20.0% — **57.5% real native**, split across 4 tiers (§3) |
| **Protected floor** | 8% of every batch, **native-Indic only** (§6) |
| **Anneal reserve** | 0.200T held back (§6) |
| **Validation** | **49/49 checks pass** (§11) · [`results/computed_output.txt`](results/computed_output.txt) |
| **Proxy programme** | 5 experiments, $10,743 = **1.70%** of the flagship run · **P4 and P5 executed** at small scale (§10) |

---

## Contents

| § | Section | Rubric criterion it answers |
|---|---|---|
| [1](#1-model-family-and-token-budget) | Model family & token budget | defended budget |
| [2](#2-capability-mixture-vs-real-supply) | Mixture vs. real supply | **share for every lane, sized against real supply** |
| [2.3](#23-named-datasets-and-the-benchmark-each-lane-defends) | Named datasets & benchmark map | **every lane tied to a real source and the benchmark it defends** |
| [3](#3-indic-lane--the-four-tier-split) | Indic four-tier split | **verified / unverified / translated / synthetic** |
| [4](#4-curriculum--five-stages) | Curriculum & stages | agentic, reasoning, long-context slots |
| [5](#5-difficulty-and-reasoning-length-bands) | Difficulty bands | **concrete example at each level** |
| [6](#6-protected-floor-and-anneal-reserve) | Floor & reserve | **fixed floor, declared reserve** |
| [7](#7-fertility--what-the-indic-budget-buys) | Fertility | carried from Session 3 |
| [8](#8-sensitivity--what-breaks-first) | Sensitivity | robustness under bad estimates |
| [9](#9-cleaning-priority--aimed-at-the-starved-slots) | Cleaning priority | **cleaning aimed at starved slots** |
| [10](#10-proxy-experiments--the-plan-as-a-testable-hypothesis) | Proxy experiments | **the plan as a testable hypothesis** |
| [11](#11-validation--4949-checks-pass) | Validation | internal consistency |
| [12](#12-one-page-defence) | One-page defence | — |

**Prior work:** [Session 3 — data & tokenizer](https://tourmaline-longma-0a9e58.netlify.app/) · [Session 4 — cleaning pipeline](https://merry-banoffee-08ed4e.netlify.app/)

---

## 1. Model family and token budget

| Model | Kind | Total | Active | tok/active | Tokens | Role |
|---|---|---|---|---|---|---|
| V5-Proxy-1B | dense | 1B | 1B | 20 | 0.020T | mixture ablation |
| V5-Proxy-3B | dense | 3B | 3B | 20 | 0.060T | mixture ablation |
| V5-Seed-8B | dense | 8B | 8B | 267 | 2.14T | growth seed / data gate |
| **V5-Base** | **MoE** | **120B** | **15B** | **267** | **4.00T** | **V4-parity flagship** |
| V5-Max | MoE | 200B | 20B | 267 | 5.34T | stretch, if collection over-delivers |

**Budget derivation** — computed in [`src/erav5/budget.py`](src/erav5/budget.py), not chosen:

| Quantity | Value |
|---|---|
| Cluster | 256 GPU × 90 d @ 40% MFU → 7.875×10²³ FLOPs |
| Feasible tokens (`C / 6·N_active`) | 8.75T |
| **Planned tokens** | **4.00T** — compute headroom **2.19×** |
| Chinchilla floor (20 tok/active-param) | 0.30T → we overtrain **13.3×** |
| tok / active param | 267 |
| tok / total param | 33.3 |
| vs ERA V4 (1.15T) | **3.48× the data** |
| **Collection target** | **8.00T cleaned = 2.0× the train budget** |

### Why 4.0T and not 15T

Three defences, strongest last:

1. **It's what the compute buys, with margin.** 8.75T is feasible; planning to 4.0T leaves 2.19× headroom for restarts, failed runs, and the proxy programme. Planning to the ceiling is planning to fail.

2. **Collecting 8T to train on 4T lets the selector *reject*.** An OPUS-class selector delivered ~6× effective-token value in V4 *by discarding*. A selector with no surplus to discard is just a bottleneck.

3. **It makes the Indic lane honest.** This is the real argument. At 15T, a 20% Indic share needs 3.0T Indic tokens against ~0.54T of real native supply — forcing the lane to 60% translated/synthetic. At 4.0T, the same 20% needs 0.80T, and **real native supply covers 57.5% of it.** My earlier draft's uncomfortable 40/60 native/non-native split was never a data problem. It was an artefact of an inflated budget.

---

## 2. Capability mixture vs. real supply

Every lane sized against countable supply. `ep` = epochs over unique tokens. `syn%` = share that must be **generated** because real supply cannot reach it.

| Lane | Share | Tokens | Unique supply | ep | Real | Synthetic | syn% | loss% | Supervised |
|---|---|---|---|---|---|---|---|---|---|
| English web & knowledge | 30.0% | 1.200T | 3.780T | 0.32× | 1.200T | — | 0% | 100% | 1.200T |
| Code | 22.0% | 0.880T | 0.910T | 0.97× | 0.880T | — | 0% | 100% | 0.880T |
| Math & science | 12.0% | 0.480T | 0.250T | 1.92× | 0.480T | — | 0% | 100% | 0.480T |
| Indic — unverified native | 7.0% | 0.280T | 0.450T | 0.62× | 0.280T | — | 0% | 100% | 0.280T |
| Agentic | 6.0% | 0.240T | 0.050T | 1.50× | 0.075T | 0.165T | **69%** | **35%** | **0.084T** |
| Indic — verified native | 4.5% | 0.180T | 0.090T | 2.00× | 0.180T | — | 0% | 100% | 0.180T |
| Indic — translated | 4.5% | 0.180T | 0.220T | 0.82× | 0.180T | — | 0% | 100% | 0.180T |
| Indic — synthetic | 4.0% | 0.160T | — | — | — | 0.160T | **100%** | 100% | 0.160T |
| Reasoning (long CoT) | 4.0% | 0.160T | 0.030T | 1.50× | 0.045T | 0.115T | **72%** | 100% | 0.160T |
| Forums & code-mix | 3.0% | 0.120T | 0.350T | 0.34× | 0.120T | — | 0% | 100% | 0.120T |
| Other multilingual | 3.0% | 0.120T | 1.500T | 0.08× | 0.120T | — | 0% | 100% | 0.120T |
| **Total** | **100%** | **4.000T** | | | **3.560T** | **0.440T** | **11.0%** | 96.1% | **3.844T** |

**Real tokens: 3.560T (89.0%). Generated: 0.440T (11.0%).**

### 2.1 The loss-mapping correction

The `loss%` column is the one most plans omit, and it's where the sharpest reviewer question lands.

In agentic trajectories, **tool observations are context-only — zero loss.** Only model-generated tokens (plans, code patches, JSON tool calls) receive gradient. Training on tool responses teaches the model to *hallucinate observations*, which is the worst failure mode an agent has — this mechanism is executed and confirmed at small scale, not just asserted (§10, P5): masking the tool-output loss took a model's tendency to fabricate a withheld tool response from 73.3% down to 0.0% on never-seen inputs, at zero cost to correctly using a genuinely supplied one.

So the agentic lane's honest size is not 0.240T. It is **0.084T of supervised tokens** — 35% of its nominal share. Any plan quoting "6% agentic" without this correction overstates that lane's real training signal by nearly 3×. Across the budget, 4.000T of tokens carry **3.844T (96.1%)** of loss.

### 2.2 Starved lanes — declared in the open

Three lanes cannot be filled from real data. Each is named, and each is paired with a **verification mechanism** that makes generation defensible rather than wishful:

| Lane | Real coverage | Must generate | Verification gate |
|---|---|---|---|
| **Agentic** | 31.2% | 0.165T | **Execution-checked** — sandbox reports task success or the trajectory is discarded |
| **Reasoning** | 28.1% | 0.115T | **Answer-checked** — final answer must verify and the derivation must re-execute |
| **Indic synthetic** | 0% | 0.160T | **COMET + round-trip + ~1% paid native audit** per batch |

The principle: a majority-synthetic lane is defensible **only** when every generated token passes an independent pass/fail gate that was not itself generated. Answer-checking and execution are such gates. "It reads fluently" is not — which is precisely why the **English lane is 0% synthetic** despite synthetic prose being the cheapest thing on this list to produce.

The agentic lane's 31.2% real coverage is **`Public agent trajectories (ToolBench/WebArena-class)`** — 0.050T unique, execution-unverified as collected, `[PUB]` in `config.py`'s inventory — topped up by **`Own sandbox self-play (terminal/browser/Indian-stack)`**, `[EST]`, which is where the execution-checking gate above actually applies. The reasoning lane's 28.1% is **`Public worked-solution / CoT corpora`** — 0.030T unique, `[PUB]` — topped up by **`RLVR-verified distilled long CoT`**, `[EST]`, gated by the answer-checking mechanism above. Full inventory with license and provenance tags: [`src/erav5/config.py`](src/erav5/config.py).

### 2.3 Named datasets and the benchmark each lane defends

Every lane's real inventory source(s) and the benchmark its share is sized to win — so a reviewer can trace *why* a number is what it is back to both a countable supply and a target, not just one or the other.

| Lane | Real inventory source(s) | Target benchmark(s) |
|---|---|---|
| English web & knowledge | FineWeb-Edu/DCLM-grade web, `.in` news/gov/courts/Hansard, books/long-form, Wikipedia | MMLU, held-out perplexity |
| Code | Stack-v2 permissive (deduped, lint/exec-gated), PRs/issues/commit histories, notebooks | HumanEval, MBPP, SWE-bench (leakage risk, §8) |
| Math & science | FineMath-grade + OpenWebMath, arXiv/PMC full text, NCERT/JEE/UPSC prep | GSM8K, MATH, in-house JEE/UPSC-style eval (§5 Band-2 example) |
| Indic — verified/unverified native | Sangraha-verified core, OCR'd books/news, legacy-font rescue, FineWeb2-Indic, Parliament/judgment records, ASR transcripts | MILU, IndicXTREME, IndicGLUE (per-language accuracy) |
| Indic — translated | BPCC + Samanantar parallel corpus, Sangraha MT-flagged portion | FLORES-200 Indic pairs, IN22 |
| Indic — synthetic | Generated (structure-preserving translation, transliteration doubles, distilled CoT) — backstops the tiers above | Same as verified/unverified — synthetic exists to cover where verified is thin, evaluated identically |
| Agentic | `Public agent trajectories (ToolBench/WebArena-class)`, `Own sandbox self-play` | BFCL function-call accuracy, τ²-bench task success |
| Reasoning | `Public worked-solution/CoT corpora`, `RLVR-verified distilled long CoT` | BBH, AIME-style held-out set (harder/longer chains than the math lane's GSM8K/MATH) |
| Forums & code-mix | Forums / romanized social (Hinglish, Tanglish) | GLUECoS / LinCE (code-mixed evaluation) |
| Other multilingual | CulturaX / CC multilingual (non-Indic) | FLORES-200 non-Indic pairs, held-out multilingual perplexity |
| Long-context (curriculum property, §4.3) | Naturally-long documents drawn from the lanes above (repos, judgments, OCR books, long-form English, full trajectories) | RULER@256K, needle-in-a-haystack retrieval |

---

## 3. Indic lane — the four-tier split

**20.0% of budget = 0.800T tokens.** No single headline number; tiers stated separately as the assignment requires.

| Tier | Tokens | Unique supply | Epochs | % of budget | % of Indic lane |
|---|---|---|---|---|---|
| **Verified native** | 0.180T | 0.090T | 2.00× | 4.5% | 22.5% |
| **Unverified native** | 0.280T | 0.450T | 0.62× | 7.0% | 35.0% |
| **Translated** | 0.180T | 0.220T | 0.82× | 4.5% | 22.5% |
| **Synthetic** | 0.160T | — | — | 4.0% | 20.0% |

| | |
|---|---|
| **NATIVE** (verified + unverified) | **0.460T = 57.5%** of the Indic lane |
| **NON-NATIVE** (translated + synthetic) | 0.340T = 42.5% |
| synthetic / native ratio | **0.35×** (cap 1.0×) — inside the 30–50% safe band |

**Tier definitions and sources:**

- **Verified native** — human-authored, script-native, passed the per-language edu classifier **and** ~1% paid native-speaker audit. Sangraha-verified core (42B) + OCR'd books/news with confidence gates (30B) + legacy-font rescue from KrutiDev/Shree-Lipi/Bamini (18B). *Sangraha's "251B" headline is ~65% machine-translated and is **excluded** from this tier.*
- **Unverified native** — script-native, not individually audited: FineWeb2-Indic + broader Sangraha pool (280B), Parliament/judgments/gov records (90B), ASR-mined transcripts (80B).
- **Translated** — BPCC + Samanantar parallel corpus (60B), Sangraha's MT-flagged portion (160B).
- **Synthetic** — structure-preserving translation into 14 languages, transliteration doubles (script↔romanized), distilled CoT re-rendered in Indic.

**Discipline:**

- Synthetic is capped as a **ratio to native**, not a fixed number. If native supply comes in lower, the synthetic ceiling falls automatically (see §8).
- Verified native runs **exactly 2.0 epochs total, including the anneal** — the anneal's verified-native component is the *deferred second pass*, carved out of the lane budget, **not** additional exposure. Getting this wrong silently pushes the scarcest tier to ~3×; [`validate.py`](src/erav5/validate.py) asserts on it.
- Translated tokens carry breadth (script coverage, code-switch robustness), not depth — their share **falls from 5.6% in S1 to 2.0% in S4**, so translationese artefacts are diluted exactly where LR is lowest and learning is stickiest.

---

## 4. Curriculum — five stages

| Stage | Frac | Tokens | Context | Span | Purpose |
|---|---|---|---|---|---|
| **S0 Warmup** | 3.75% | 0.150T | 8K | 0 → 0.150T | embedding + **router** warmup; mix held at S1 proportions |
| **S1 Foundation** | 61.25% | 2.450T | 8K | 0.150 → 2.600T | web-heavy; language, world knowledge, syntax |
| **S2 Skill ramp** | 22.50% | 0.900T | 32K | 2.600 → 3.500T | code/math/reasoning up; 8K→32K |
| **S3 Long context** | 7.50% | 0.300T | 256K | 3.500 → 3.800T | 32K→256K on naturally-long docs only |
| **S4 Anneal** | 5.00% | 0.200T | 256K | 3.800 → 4.000T | Tier-A reserve only; LR → 0 |

### 4.1 Stage × lane reconciliation

Global shares are a *weighted average* of stage mixes. If the stages don't integrate back to the global mixture, the plan is internally inconsistent. **S1 is not hand-tuned — it is solved** ([`curriculum.solve_s1()`](src/erav5/curriculum.py)) so integration is exact by construction.

| Lane | S0 | S1 | S2 | S3 | S4 | Integrates to | Target | Drift |
|---|---|---|---|---|---|---|---|---|
| english | 38.0% | 34.4% | 19.0% | 30.0% | 20.0% | 30.00% | 30.0% | 0.00pp |
| code | 17.0% | 19.0% | 31.0% | 29.0% | 12.0% | 22.00% | 22.0% | 0.00pp |
| math | 9.0% | 9.9% | 18.0% | 11.0% | 15.0% | 12.00% | 12.0% | 0.00pp |
| indic_verified | 3.0% | 3.5% | 4.0% | 6.0% | **18.0%** | 4.50% | 4.5% | 0.00pp |
| indic_unverified | 7.0% | 6.7% | 7.0% | 10.0% | 6.0% | 7.00% | 7.0% | 0.00pp |
| indic_translated | 5.5% | 5.6% | 3.0% | 1.5% | 2.0% | 4.50% | 4.5% | 0.00pp |
| indic_synthetic | 3.5% | 4.4% | 4.0% | 2.0% | 3.0% | 4.00% | 4.0% | 0.00pp |
| agentic | 3.0% | 4.6% | 8.5% | 7.0% | **13.0%** | 6.00% | 6.0% | 0.00pp |
| reasoning | 2.0% | 3.5% | 4.5% | 3.0% | **11.0%** | 4.00% | 4.0% | 0.00pp |
| forums | 6.0% | 4.3% | 0.5% | 0.0% | 0.0% | 3.00% | 3.0% | 0.00pp |
| multilingual | 6.0% | 4.3% | 0.5% | 0.5% | 0.0% | 3.00% | 3.0% | 0.00pp |

### 4.2 Gradient-safe blending bands

V4 recorded a **~150× gradient-norm spike** when the Hindi share was raised abruptly. Every transition is therefore **ramped, not stepped**, at a rate limit of **12pp per 100B tokens**. Ramps **straddle** the boundary (half in the outgoing tail, half in the incoming head) — otherwise the S3→S4 ramp alone would consume 70% of the anneal and destroy the cooldown.

The ramp-vs-step mechanism itself is executed and confirmed at small scale (§10, P4) — real text on a real GPU reproduces the shock (6.04× at an abrupt transition) and shows it collapsing to ~1.5–1.8× with even a short ramp, with little further gain from lengthening it. The rate limit above (12pp/100B) is not yet the number that small-scale evidence sets — it stands pending the real 3B/60B P4 run, which the small-scale result now points toward testing a shorter band first.

| Transition | Worst lane | Δ | Ramp | Split | Rate |
|---|---|---|---|---|---|
| S0 → S1 | english | −3.6pp | 30.2B | 15.1B + 15.1B (1% of S1) | 12.0pp/100B |
| S1 → S2 | english | −15.4pp | 128.1B | 64.0B + 64.0B (7% of S2) | 12.0pp/100B |
| S2 → S3 | english | +11.0pp | 91.7B | 45.8B + 45.8B (15% of S3) | 12.0pp/100B |
| S3 → S4 | code | −17.0pp | 141.7B | 70.8B + 70.8B (35% of S4) | 12.0pp/100B |

### 4.3 Long context — a curriculum property, not a token lane

RULER@256K is targeted by **selecting naturally-long documents already inside existing lanes**. No separate budget slot, because concatenating unrelated short documents to fill 256K teaches the model the wrong thing.

| Source lane | Document type | Trains |
|---|---|---|
| code | whole repos concatenated by import graph | cross-file dependency reasoning |
| indic_unverified | judgments, Hansard, gov reports | long-document Indic comprehension |
| indic_verified | OCR'd full-length books | sustained native narrative |
| english | books, legal/technical long-form | RULER@256K parity |
| agentic | full multi-step trajectories, unsplit | long tool-history management |

Short-document lanes (forums 0.0%, multilingual 0.5% in S3) are excluded from the long-context stage **by construction**.

---

## 5. Difficulty and reasoning-length bands

| Band | Steps | Reasoning tokens | Concrete example | Schedule |
|---|---|---|---|---|
| **0 — Direct recall** | 0 | <32 | *"Odisha ki rajdhani kya hai?" → "Bhubaneswar."* | S0–S1 dominant, **never removed** |
| **1 — Short CoT** | 1–3 | 32–256 | *Sam has 3 apples, buys 2, gives 1 away. 3+2−1 = 4.* | S1 primary, carried at reduced weight |
| **2 — Medium CoT** | 4–8 | 256–1K | *JEE-mains projectile: pick formula → substitute → carry units → sanity-check magnitude → state answer.* | ramps in through S2 |
| **3 — Long CoT / tool-augmented** | 8–20 | 1K–8K | *Agentic: "Book an IRCTC ticket under ₹800, verify the PNR, retry once on failure" — plan, call, read error, replan.* | S2 late + S3; concentrated in S4 |
| **4 — Extended / cross-document** | 20+ | 8K–64K | *Read a 40-page judgment + a prior ruling + a Hansard excerpt; identify which precedent controls and why.* | S3–S4 only — needs 256K context to exist first |

**Effort dial** — the "controllable reasoning depth" objective. Simple problems must stay cheap:

| Level | Thinking budget | Band mix |
|---|---|---|
| low | <128 tokens | B0 60% / B1 40% |
| medium | 128–1K | B1 45% / B2 55% |
| high | 1K–8K | B2 35% / B3 65% |
| ultra | 8K–64K | B3 40% / B4 60% |

**Band 0 is never dropped.** A model trained only on long traces loses the ability to answer briefly — a real regression, not a cosmetic one.

---

## 6. Protected floor and anneal reserve

### 6.1 The floor

> **8% of every batch — 41 of 512 sequences — drawn from NATIVE Indic only (verified + unverified).**

V4 ran an 8% always-on Indic channel. I keep the percentage and **tighten the definition**: translated and synthetic tokens do **not** count.

This matters because English-tuned quality proxies systematically reject Indic script, code-mixing, and local content — so an unprotected selector starves the lane. But a floor defined on the *whole* Indic lane is a fake floor: the selector satisfies "8% Indic" entirely with cheap generated text and never touches the scarce native tiers. That is the exact language erasure the floor exists to prevent.

| Stage | Native Indic available | |
|---|---|---|
| S0 Warmup | 10.0% | ← **binding constraint** |
| S1 Foundation | 10.2% | |
| S2 Skill ramp | 11.0% | |
| S3 Long context | 16.0% | |
| S4 Anneal | 24.0% | |

**Satisfiable in every stage, slack +2.0pp.** Floor sequences are exempt from selector rejection (individual documents still pass PII/toxicity screening), so no gradient step is ever taken without Indian-language signal.

### 6.2 The anneal reserve

**0.200T = 5.0% of budget, held BACK from the lane budgets — not added on top.**

| Component | Share | Tokens | Rationale |
|---|---|---|---|
| Instruction-dense general text | 28% | 0.056T | matches downstream SFT distribution |
| **Verified-native Indic (deferred 2nd epoch)** | 22% | 0.044T | the 2nd pass is *spent* here, not added; total exposure stays 2.0× |
| Answer-verified math / worked solutions | 18% | 0.036T | OLMo 2 precedent: small late reserve moved grade-school math 24% → 67% |
| Execution-verified agentic trajectories | 16% | 0.032T | sandbox-successful traces only; loss on model tokens only |
| Long-CoT Band 3–4 reasoning | 11% | 0.022T | effort-dial supervision lands where LR is lowest |
| Safety + long-context refresh | 5% | 0.010T | 12+ language refusals incl. romanized attacks; one 256K pass so long-context survives the cooldown |

**Explicitly excluded:** benchmark test items, eval-adjacent paraphrases, **canary strings**, unverified synthetic CoT.

Canaries live *only* in the private eval suites. Placing them in the anneal would be deliberate contamination — it inverts their purpose, which is to *detect* leakage.

---

## 7. Fertility — what the Indic budget buys

Carried unchanged from [Session 3](https://tourmaline-longma-0a9e58.netlify.app/): **vocab 196,608 (3·2¹⁶)**, frozen before pretraining — the one irreversible decision.

**Weighted Indic fertility: 1.7199 tokens/word** (target ≤ 1.75), weighted by internet-user share across 13 first-class languages.

Same **0.800T Indic token budget**, in *words the model actually learns*:

| Tokenizer | Fertility | Words | vs English-centric BPE |
|---|---|---|---|
| **ERAV5 target** | 1.72 | **0.465T** | **2.62×** |
| Gemma 4 (262K) | 2.20 | 0.364T | 2.05× |
| English-centric BPE | 4.50 | 0.178T | 1.00× |

Tokens are the bill; words are what the model learns. Same budget, **2.6× the language**.

**Session-2 balance metric** — `Score = 1000 / (X_max − X_min)`:
X_max = 2.15 (Malayalam), X_min = 1.40 (English), spread 0.75 → **score 1333.3**. The metric rewards *balance*, not a low average — which is why Dravidian targets are relaxed per-word (agglutinative morphology packs more meaning into one word) while held per-byte.

---

## 8. Sensitivity — what breaks first

The **90B verified-native estimate is the most fragile input in this document.** Every other lane has slack. Targets are fixed by the mixture; what moves under a supply shock is *how much of each target is backed by real tokens*.

| Scenario | vn epochs | Real native | % of Indic | Indic gap | Overall real% | Floor |
|---|---|---|---|---|---|---|
| Baseline | 2.00× | 0.460T | 57.5% | 0.160T | 89.0% | OK |
| Verified native −33% | 2.00× | 0.401T | 50.1% | 0.219T | 87.5% | OK |
| Verified native −50% (worst credible) | 2.00× | 0.370T | **46.2%** | 0.250T | 86.8% | OK |
| Code −30% after license filter | 2.00× | 0.460T | 57.5% | 0.160T | 89.0% | OK |
| Unverified Indic −40% (lang-ID mislabels) | 2.00× | 0.460T | 57.5% | 0.160T | 89.0% | OK |
| All Indic native −40% (systemic) | 2.00× | 0.388T | 48.5% | 0.232T | 87.2% | OK |

**Stated responses — the plan bends, it does not break:**

- **Verified native misses:** hold exposure at 2.0× (do *not* raise epochs). Backfill from unverified native, which sits at 0.62 epochs with **~0.17T unused headroom**. Synthetic does **not** expand to fill the gap — the parity cap is a *ratio*, so shrinking native automatically shrinks the synthetic ceiling.
- **Code misses:** move to 1.3–1.5× epochs before considering synthetic code. Repetition is safer than generation in a lane where SWE-bench leakage is the dominant risk and generated code has no correctness signal independent of the tests we also generated.
- **Unverified Indic misses:** most slack of any lane — a 40% cut is absorbed at 0.62 epochs. If both native tiers shrink, the Indic headline drops below 20% to whatever native supports. **The share follows the supply, not the reverse.**
- **Agentic misses:** no change. Designed around self-play from the start; P3 gates whether self-play works at all, which is the real risk.

The floor holds in **every** scenario — because it's defined as a share of the *batch*, not of a supply pool that can evaporate.

---

## 9. Cleaning priority — aimed at the starved slots

Applying my [Session-4 pipeline](https://merry-banoffee-08ed4e.netlify.app/) yields to the Session-5 mixture: how many **raw** tokens must enter the pipeline to produce the clean tokens each lane needs, including the 2× collection headroom.

| Lane | Pipeline yield | Clean needed | Collection target | **Raw required** | Starved |
|---|---|---|---|---|---|
| english | 22% | 1.200T | 2.400T | 10.909T | |
| code | 31% | 0.880T | 1.760T | 5.677T | |
| math | 18% | 0.480T | 0.960T | 5.333T | |
| **indic_verified** | **9%** | 0.180T | 0.360T | **4.000T** | |
| indic_unverified | 26% | 0.280T | 0.560T | 2.154T | |
| forums | 14% | 0.120T | 0.240T | 1.714T | |
| multilingual | 24% | 0.120T | 0.240T | 1.000T | |
| indic_translated | 55% | 0.180T | 0.360T | 0.655T | |
| agentic | 42% | 0.075T | 0.150T | 0.357T | **YES** |
| reasoning | 35% | 0.045T | 0.090T | 0.257T | **YES** |
| indic_synthetic | — | — | — | — | **YES** (generated) |
| **Total raw intake** | | **3.560T** | **7.120T** | **32.06T** | |

Verified-native Indic has a **9% yield** — OCR confidence gates plus the native-speaker audit reject hard. It takes **4.0T of raw intake to land 0.18T of usable tokens**. That is the real reason the tier is scarce: not that the text doesn't exist, but that *verifying* it is expensive.

**Where to point the remaining cohort cleaning effort** (marginal value of one more cleaned token; floor-eligible lanes weighted 2×):

| Rank | Lane | Score | Supply headroom | Epochs | |
|---|---|---|---|---|---|
| 1 | reasoning | 5.33 | 0.19× | 1.50 | |
| 2 | agentic | 4.80 | 0.21× | 1.50 | |
| 3 | **indic_verified** | 4.00 | 0.50× | 2.00 | floor lane — counts double |
| 4 | math | 1.92 | 0.52× | 1.92 | |
| 5 | indic_unverified | 1.24 | 1.61× | 0.62 | floor lane — counts double |
| 6 | code | 0.97 | 1.03× | 0.97 | |

English, forums, and multilingual are **deprioritised** — 3×, 3×, and 12× supply headroom respectively. Cleaning more English is effort spent where it changes nothing.

---

## 10. Proxy experiments — the plan as a testable hypothesis

Every ratio here is a hypothesis until a cheap experiment tests it. Each proxy states what would **refute** the plan, not only what would confirm it — a decision rule that can only fire one way is not a test.

| ID | Scale | Tokens/arm | Arms | GPU-h | Cost | Question | Status |
|---|---|---|---|---|---|---|---|
| **P5** | 1B | 20B | 2 | 169 | $421 | Loss-masking on tool observations | **executed at small scale — see below** |
| **P2** | 1B | 20B | 2 | 169 | $421 | Protected floor necessity | pre-registered |
| **P3** | 1B | 20B | 2 | 169 | $421 | Agentic verification gate | pre-registered |
| **P1** | 3B | 60B | 2 | 1,517 | $3,792 | Indic synthetic parity cap | pre-registered |
| **P4** | 3B | 60B | 3 | 2,275 | $5,688 | Mixture-transition gradient stability | **executed at small scale — see below** |
| | | | | **4,297** | **$10,743** | **= 1.70% of the flagship run (~$632K)** | |

**Run order: P5 → P2 → P3 → P1 → P4.** P5 is cheapest and its result changes how every other agentic number is counted, so it goes first — both P4 and P5 have now been run out of order at small scale, ahead of P2/P3/P1, because their mechanisms (a scheduling question and a loss-masking question, not language-quality questions) are the two a small synthetic proxy can actually test without curated real-language data or an eval harness.

<details>
<summary><b>P5 · Loss-masking on tool observations</b> (1B, 20B/arm) — <b>executed at small scale, numbers below</b></summary>

- **H:** zero-loss masking of tool outputs prevents observation hallucination without costing task success.
- **A:** loss on model-generated tokens only · **B:** loss on the full trajectory including tool responses.
- **Metric:** rate of fabricated tool observations in held-out rollouts; BFCL.
- **Rule:** A must cut fabricated-observation rate by **≥50% at no BFCL cost**. If masking costs task success, the 35% loss-fraction assumption in §2.1 is wrong and the agentic lane must be re-sized.

**What was actually run:** a synthetic tool-use trajectory small enough for a single consumer GPU — single-digit addition via a fixed template `Q{a}{b} C{a}{b} O{sum} A{parity}`, where `O{sum}` is the tool's (environment-provided) response and `A{parity}` is the model's answer, which requires correctly reading that response. A 152,832-param transformer, 2 arms × 3 seeds, on an RTX 3050 Laptop GPU. Full setup, script, and raw output: [`proxy_runs/p5_loss_masking/`](proxy_runs/p5_loss_masking/results.md).

| Arm | Fabrication (train) | Fabrication (held-out) | Task success (train) | Task success (held-out) |
|---|---:|---:|---:|---:|
| A — masked | **0.00** | **0.00** | 1.00 | 1.00 |
| B — full loss | 1.00 | **0.733** | 1.00 | 1.00 |

Every one of 3 seeds agrees: Arm A never reproduces the tool's output when it isn't given one; Arm B reproduces it perfectly on seen combos and **73.3% of the time on combos it never saw during training** — it has generalized the underlying function well enough to confidently fabricate a plausible, often-correct answer instead of needing the tool. Task success is perfect for both arms — masking costs nothing. Applying the rule above: held-out fabrication drops from 73.3% to 0.0%, a 100% reduction, comfortably clearing the plan's own 50% bar, at zero task-success cost. **The hypothesis is confirmed, decisively, by all 3 seeds independently.** This is single-digit addition in an 11-character trajectory, not real agent tool-use or the real BFCL/τ²-bench metrics — the *magnitude* on real trajectories still needs the 1B/20B run in §10 — but the *mechanism* behind §2.1's 0.084T-supervised-tokens claim is no longer an assumption; it has been shown to work exactly as claimed on a real executed run.
</details>

<details>
<summary><b>P2 · Protected floor necessity</b> (1B, 20B/arm)</summary>

- **H:** without a native-only always-on floor, Indic capability drifts down during the code/math-heavy S2 phase.
- **A:** 8% native-Indic floor every batch · **B:** same global Indic share, no batch-level floor.
- **Metric:** MILU (Hindi + one Dravidian) at **5 checkpoints — the curve shape is the evidence, not the endpoint.** Secondary: HumanEval/GSM-mini to confirm no tax on dominant lanes.
- **Rule:** keep the floor if B shows a mid-training MILU dip **≥3 pts** that A avoids. **If both curves are flat, drop the floor** — unnecessary complexity that costs selector flexibility.
</details>

<details>
<summary><b>P3 · Agentic verification gate</b> (1B, 20B/arm)</summary>

- **H:** execution-verified sandbox trajectories beat imitation-only public agent data at equal token count.
- **A:** sandbox self-play, kept only on task success · **B:** public trajectories, no verification gate.
- **Metric:** BFCL function-call accuracy; τ²-bench task success.
- **Rule:** A must clear B by **≥5 pts on both**. If not, the generation pipeline is the defect to fix before the 6% agentic share scales — **not** a reason to cut the lane.
</details>

<details>
<summary><b>P1 · Indic synthetic parity cap</b> (3B, 60B/arm)</summary>

- **H:** capping synthetic Indic at ≤1.0× native beats a synthetic-heavy arm on native-quality metrics at matched compute.
- **A:** plan mixture (0.35× ratio) · **B:** synthetic doubled (0.70×), translated held constant.
- **Metric:** MILU per Tier-1 language; FLORES chrF++ (native vs synthetic-heavy runs diverge here first); 200-sample native-speaker audit for translationese.
- **Rule:** adopt the cap only if A beats B by **≥2.0 pts mean MILU with no chrF++ regression**. **If B wins, the cap rises** — the plan was too conservative and we're leaving Indic capability on the table. If neither separates, drop the cap as unjustified machinery.
</details>

<details>
<summary><b>P4 · Mixture-transition gradient stability</b> (3B, 60B/arm, 3 arms) — <b>executed at small scale, numbers below</b></summary>

- **H:** ramping lane-share changes across a ≥100B-token band prevents the ~150× gradient-norm spike V4 recorded.
- **A:** 100B linear ramp · **B:** abrupt step at the boundary · **C:** 20B ramp (is the band over-specified?)
- **Metric:** max gradient-norm multiplier over the transition; loss-spike count.
- **Rule:** keep the 100B band only if B spikes **≥5×** *and* C also spikes. **If 20B suffices, shorten the band** and reclaim schedule — arm C exists specifically to catch us over-engineering.

**What was actually run:** the 3B/60B-per-arm version above needs a cluster. What a single consumer GPU can do instead is test the same *mechanism* — does an abrupt lane-share change shock the gradient, and does ramping fix it — at increasing rigor, in two passes. Full setup, scripts, and raw traces: [`proxy_runs/p4_gradient_stability/`](proxy_runs/p4_gradient_stability/).

**Pass 1** (818K-param transformer, synthetic Markov-chain "lanes", single seed, 3 ramp lengths — [results.md](proxy_runs/p4_gradient_stability/results.md)):

| Arm | Ramp (steps) | Grad-norm multiplier | Loss spikes |
|---|---:|---:|---:|
| B — abrupt step | 0 | 2.84× | 11 |
| C — short ramp (∝ 20B) | 60 | 1.29× | 0 |
| A — long ramp (∝ 100B) | 300 | 1.11× | 0 |

**Pass 2, deepened** (2.73M-param transformer — 3.3× larger; lane A/B are this repo's own real prose vs. real Python source, not synthetic chains; 8 ramp lengths swept from 0–300 steps; 3 seeds per point reported as mean ± std — [results_v2.md](proxy_runs/p4_gradient_stability/results_v2.md)):

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

Real text produces a bigger, more realistic shock than the synthetic chains did (6.04× vs 2.84×), which strengthens confidence the mechanism itself is real. Applying the rule above literally: B clears the 5× bar convincingly; every non-zero ramp tested — down to 20 steps — spikes 0 loss-spikes and sits at 1.5–1.8× multiplier, well under any reasonable spike bar. The curve from 20 to 300 steps is flat and noisy (per-point std often comparable to the gaps between points), not a clean monotonic decline, so the honest read is narrower and sharper than "somewhere under 100B": **nearly all of ramping's protection is captured by a short ramp, and lengthening it well past that point buys little at this scale.** The specific "100B vs 20B tokens" number for a 15B-active MoE still needs the real 3B/60B run — two independent small-scale runs agreeing on the *shape* of the answer is evidence to prioritize testing a **shorter** band as the primary candidate there, not the 100B width by default.
</details>

**Status: P4 (two passes) and P5 executed at small scale (results above); P1, P2, P3 pre-registered, not yet executed.** This document is the hypothesis. Numbers in §1–§9 stand until a proxy fires — P4's small-scale runs confirm the ramping mechanism and point toward a shorter band without setting the exact 100B-vs-20B width; P5's small-scale run confirms the loss-masking mechanism decisively (100% reduction in fabrication, zero task-success cost, unanimous across 3 seeds), which is real support for §2.1's 0.084T-supervised claim, though the real 1B/20B run is still what validates it on genuine agent trajectories.

---

## 11. Validation — 49/49 checks pass

The plan is machine-checked, so it cannot silently drift out of consistency when a number changes. [`src/erav5/validate.py`](src/erav5/validate.py) asserts:

- **Arithmetic** — mixture shares sum to 1.0; all 5 stage mixes sum to 1.0; stage fractions sum to 1.0
- **Integration** — stage schedule reproduces the global mixture (max drift **0.00pp**, tolerance 0.5pp)
- **Repetition** — every lane within its epoch cap *and* below the 4-epoch free-repetition ceiling
- **No silent generation** — every synthetic gap is in a lane declared generable, with volume named
- **Indic** — synthetic ≤ parity with native (0.35× vs 1.0× cap); Indic lane is majority native (57.5%)
- **The epoch trap** — verified-native anneal component is a *subset* of the lane budget (44B ≤ 180B); total exposure 2.00× vs a 2.0× cap
- **Floor** — satisfiable in every stage (binding: S0 at 10.0% vs 8% floor)
- **Reserve** — shares sum to 1.0; reserve size equals S4 stage size exactly
- **Gradient safety** — all 4 transitions within 12pp/100B; all ramps fit inside their stages
- **Fertility** — weighted Indic 1.7199 ≤ 1.75

Full output: [`results/computed_output.txt`](results/computed_output.txt) · Machine-readable: [`results/plan.json`](results/plan.json)

---

## 12. One-page defence

| Lane | Share | Real backing | Flag |
|---|---|---|---|
| English | 30.0% | 100% real, 0.32 ep, 3.2× headroom | none |
| Code | 22.0% | 100% real, 0.97 ep | tight — 1.03× headroom |
| Math & science | 12.0% | 100% real, 1.92 ep | at epoch cap |
| Indic — verified native | 4.5% | 100% real, **2.00 ep (at cap)** | scarcest tier; 9% pipeline yield |
| Indic — unverified native | 7.0% | 100% real, 0.62 ep | most slack in the plan |
| Indic — translated | 4.5% | MT-derived, 0.82 ep | breadth only; falls to 2% by S4 |
| Indic — synthetic | 4.0% | generated, 0.35× native | inside safe band; COMET + audit gated |
| Agentic | 6.0% | 31% real / 69% generated | execution-verified; **only 0.084T supervised** |
| Reasoning | 4.0% | 28% real / 72% generated | answer-verified |
| Forums & code-mix | 3.0% | 100% real, 0.34 ep | none |
| Other multilingual | 3.0% | 100% real, 0.08 ep | 12× headroom, sampled down |

**The three claims this plan stakes itself on:**

1. **The budget is set by MoE active parameters (15B), not total (120B)** — which is why 4.0T is the right number and 15T was never available.
2. **A smaller budget makes the Indic lane more honest, not less** — 57.5% native at 4.0T versus 40% native at 15T, from identical real supply.
3. **Every lane over 50% synthetic is gated by an independent verifier** — execution, answer-checking, or paid native audit. Where no such gate exists (English prose), the synthetic share is **zero**.

---

## Repository layout

```
├── README.md                  ← this document (the submission)
├── docs/
│   └── SPECIFICATION.md       full spec, same content, standalone
├── results/
│   ├── computed_output.txt    11-section run output, 49/49 passing
│   └── plan.json              machine-readable export of every number
├── proxy_runs/
│   ├── p4_gradient_stability/ P4 executed on GPU, two passes — scripts, raw traces, results.md/results_v2.md
│   └── p5_loss_masking/       P5 executed on GPU — script, raw output, results.md
└── src/erav5/
    ├── config.py              ALL input assumptions — change one number, plan re-derives
    ├── budget.py              compute → tokens (MoE-aware: C = 6·N_active·D)
    ├── mixture.py             supply audit, epochs, synthetic gaps, loss mapping
    ├── curriculum.py          5 stages, stage×lane integration, blending bands, difficulty bands
    ├── floors.py              protected floor, anneal reserve, fertility
    ├── proxy.py               5 proxy experiments, costed
    ├── sensitivity.py         supply shocks + cleaning priority
    ├── validate.py            49 assertions
    ├── export.py              JSON dump
    └── run_plan.py            driver
```

**Provenance tags in `config.py`:** `[V4]` measured on the V4 run · `[SESS]` stated in session material · `[PUB]` published external figure · `[EST]` our estimate. Reviewers should push hardest on `[EST]`, and §8 pre-answers the ones that matter.

### Reproduce

```bash
cd src
python3 -m erav5.run_plan      # full plan, 11 sections
python3 -m erav5.export        # JSON
echo $?                        # non-zero if any assertion fails
```
