# Submission — Session 8

## Question 1 — Live app and repo

- **Live:** _(Netlify — publish directory `assignment8/site`; see [README § Deploying](README.md#deploying))_
- **Repo:** https://github.com/subramanyanaik/schoolofai_subbu/tree/main/assignment8

Date sources for all 39 mechanisms: **[DATES.md](DATES.md)**.

---

## Question 2 — What the timeline actually shows

A list spaces every entry evenly. That is its whole problem: it silently discards the one variable
that carries the story, which is **how long things took**. Eight things became visible only after
the dates were real. Every interval below is measured from the data by
[`tools/gaps.py`](tools/gaps.py), not estimated.

### 1. A mechanism can be four years early and simply be ignored

**MQA → GQA: 1,293 days (3.5 years).**

MQA lands 6 Nov 2019, right in the middle of the sparse-attention gold rush — Sparse Transformer,
Reformer, Longformer, linear attention, BigBird and Performer all appear within fourteen months
either side of it. Every one of those attacks the **compute** bill. MQA attacks the **memory**
bill, and then nothing happens for three and a half years.

Grouped by family, MQA and GQA sit adjacent and read as a tidy progression. In date order there is
a four-year hole, and *the hole is the finding*: nobody optimises a bill they are not yet paying.
In 2019 these models were not being served to millions of people, so the KV cache was not anyone's
problem. The idea was correct and simply too early.

> **So what:** the bill you can see is the bill you are currently paying. Ideas that answer the
> *next* bottleneck get published and then sit.

### 2. The 2020 approximation wave dies at a single point, and an engineer killed it

**Five approximations in nine months, then a two-year silence.**

Reformer (13 Jan 2020), Longformer (10 Apr), linear attention (29 Jun), BigBird (28 Jul),
Performer (30 Sep). Then the family goes quiet. FlashAttention arrives 27 May 2022 and makes
*exact* dense attention faster than most of the approximations that existed to avoid it.

In a family-grouped list FlashAttention is filed under "systems" and looks like a different
conversation. In date order it sits exactly at the extinction boundary. They were not beaten by a
better approximation — they were beaten by someone noticing attention was **memory-bandwidth**-bound,
not compute-bound. The entire wave had been optimising the wrong resource.

> **So what:** before you approximate something, check which resource is actually scarce. Three
> years of clever math lost to one profiling insight.

### 3. The scramble for context length is measured in days, and happened outside academia

**Position Interpolation → NTK-aware: 1 day. NTK-aware → YaRN: 64 days.**

PI is submitted 27 Jun 2023. NTK-aware scaled RoPE appears **the next day** — as a Reddit post,
with no paper, no peer review and no ablations at scale. YaRN follows nine weeks later and fixes
what both got wrong.

A list files all three under "RoPE scaling" and the internal ordering is arbitrary. The dates show
a community scramble triggered by Llama-2 shipping a 4K window — not a research programme. It also
shows that for several months the most widely deployed context-extension method in the world was a
forum post, which is precisely why it is the one date on this page I could not pin to a day.

> **So what:** deployment pressure sets the pace, and the artefact that wins is not always the one
> with a paper.

### 4. Good ideas wait about three years for the hardware, and the lag is consistent

**Delta rule → parallel training algorithm: 1,204 days (3.3 years).**

The delta rule for linear attention is published 22 Feb 2021, then goes essentially unused until
the chunkwise parallel algorithm arrives 10 Jun 2024. The mechanism was never the problem. It was
sequentially dependent, so training could not use a GPU, which made it worthless at scale no
matter how good the idea was.

As a single list entry, "DeltaNet" has one date and choosing which one is a coin flip — this is
exactly the item the session got wrong. Splitting it into the idea (2021) and the thing that made
it runnable (2024) exposes a gap almost identical to MQA's, for a related reason: not conceptual,
but engineering.

> **So what:** the distance between "this works" and "this runs on the hardware we have" is roughly
> three years, twice, independently. Budget for it.

### 5. The best ideas arrive twice, within days, from different labs

**NSA → MoBA: 2 days. Learned absolute positions → sinusoidal: 35 days.**

NSA (DeepSeek, 16 Feb 2025) and MoBA (Moonshot, 18 Feb 2025) independently propose the same thing:
learned block selection, trained natively rather than bolted on at inference. Forty-eight hours
apart. The same pattern opens the timeline, with learned positions and sinusoidal five weeks apart.

A list shows two similar entries and invites "which was first?" or "which is better?". The dates
show the question is wrong. When two labs with no contact ship the same idea in the same week, the
idea was **due** — determined by the hardware and the context lengths people wanted, not by
anyone's individual insight.

> **So what:** simultaneity means the constraint is driving, not the researcher. It is also the
> best available evidence that an idea is correct rather than lucky.

### 6. Position encoding is being deleted, monotonically, and you can see where it ends

**2017 → 2025: L<sub>max</sub>·d parameters → 0 → 0 → none at all.**

A learned table (2017; L<sub>max</sub>·d parameters and a hard length ceiling) → sinusoidal (2017;
no parameters, still added to the residual stream) → RoPE (2021; no parameters, moved *out* of the
residual stream and into the score) → ALiBi (2021; just a bias on the logit) → DroPE (2025; remove
it entirely after training and recalibrate briefly).

As a list this is a taxonomy, and every entry looks like an alternative to the others. In date
order, **every single step removes something, and none ever adds it back.** That is not a taxonomy,
it is a trend line with a visible endpoint.

> **So what:** this is the one place on the timeline where extrapolation is genuinely safe, because
> the direction has not reversed once in eight years. The endpoint is zero.

### 7. The oscillation is a control loop, and the trigger is always a benchmark

**exactness → length → memory → length → memory.**

Every swing back toward memory is triggered by an evaluation exposing what the previous swing
forgot. Long Range Arena and failed replications killed the 2020 approximations. Needle-in-a-haystack
and RULER exposed ALiBi's extrapolation as *increasing locality* rather than increasing reach, and
exposed the fixed-capacity ceiling in pure-linear models.

A list of mechanisms contains no benchmarks at all, so the cause of each turn is invisible and the
oscillation looks like fashion. On a timeline the benchmark lands between the cheap mechanism and
the correction, every time, and it starts looking like feedback.

> **So what:** this is what makes your question answerable. The swings are predictable from which
> benchmark lands next. As of April 2026, aggressive sequence-axis compression has not yet met its
> adversarial benchmark — so that is where I would expect the next correction.

### 8. Attacking both bills at once is a recent capability, not an old one

**First headline claim on both meters: April 2026.**

Era 2 attacks compute. Era 4 attacks memory. Almost nothing before 2025 bills both meters at once —
sliding window is the rare early exception, and it pays in retrieval quality. DeepSeek-V4's
CSA + HCA (26 Apr 2026) is the first to report both as headline results: 27% of the FLOPs and 10%
of the KV cache of its predecessor.

In a family-grouped list, compute methods and memory methods are separate sections, so you never
notice that for six years essentially nobody managed both.

> **So what:** if you are choosing a mechanism today, the pre-2025 ones make you pick a meter. Only
> the newest generation lets you refuse the choice.

---

## Question 2, bonus — mechanisms not covered in the session

Fifteen, all with the same pros/cons treatment as the required list. Dates are v1 arXiv submission
timestamps, verified via the arXiv API `<published>` field and cross-checked against each paper's
abstract page.

| Date | Mechanism | Source for the date |
|---|---|---|
| 1 Sep 2014 | Additive (Bahdanau) attention | [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) |
| 17 Aug 2015 | Multiplicative (Luong) dot-product attention | [arXiv:1508.04025](https://arxiv.org/abs/1508.04025) |
| 6 Mar 2018 | Relative position representations (Shaw) | [arXiv:1803.02155](https://arxiv.org/abs/1803.02155) |
| 9 Jan 2019 | Transformer-XL segment recurrence | [arXiv:1901.02860](https://arxiv.org/abs/1901.02860) |
| 13 Jan 2020 | Reformer (LSH attention) | [arXiv:2001.04451](https://arxiv.org/abs/2001.04451) |
| 28 Jul 2020 | BigBird (window + global + random) | [arXiv:2007.14062](https://arxiv.org/abs/2007.14062) |
| 30 Sep 2020 | Performer (FAVOR+) | [arXiv:2009.14794](https://arxiv.org/abs/2009.14794) |
| **27 May 2022** | **FlashAttention** | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| 27 Jun 2023 | Position Interpolation | [arXiv:2306.15595](https://arxiv.org/abs/2306.15595) |
| **12 Sep 2023** | **PagedAttention (vLLM)** | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) |
| 21 Feb 2024 | LongRoPE | [arXiv:2402.13753](https://arxiv.org/abs/2402.13753) |
| 10 Apr 2024 | Infini-attention | [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) |
| **7 Oct 2024** | **Differential Transformer** | [arXiv:2410.05258](https://arxiv.org/abs/2410.05258) |
| 18 Feb 2025 | MoBA (Mixture of Block Attention) | [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) |
| 30 Oct 2025 | Kimi Delta Attention (per-channel gating) | [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) |

**The four that change the story rather than lengthen the list:**

- **FlashAttention (27 May 2022)** — the reason the 2020 efficient-transformer wave died. Without
  it on the timeline, the two-year silence after Performer has no explanation. It is also the only
  mechanism on the whole page with *no* quality cost, which makes it the one honest "just use this".
- **PagedAttention (12 Sep 2023)** — the honest asterisk on 2024's "1M context" claims. Measured
  60–80% of allocated KV memory was simply wasted on fragmentation. A meaningful share of that era's
  context wins were allocation strategy, not attention math, and a page about attention that omits
  this is quietly taking credit for someone else's work.
- **Differential Transformer (7 Oct 2024)** — the only entry that attacks the **quality** bill
  rather than either cost bill, at roughly 2× the attention compute. It is the counterexample that
  makes the "every mechanism is a trade" claim precise: here the trade runs the other way.
- **NoPE ([arXiv:2305.19466](https://arxiv.org/abs/2305.19466), 31 May 2023)** — noted on the DroPE
  entry rather than as its own row. It showed causal decoders can infer position from the mask alone
  two and a half years before DroPE, which makes DroPE a much less surprising result than it first
  appears.

One more worth flagging, which is an *architecture* detail rather than a mechanism: **gpt-oss**
(5 Aug 2025, [model card](https://arxiv.org/abs/2508.10925)) ships a **learned per-head attention
sink logit** in the softmax denominator. StreamingLLM's 2023 cache hack became a trained parameter.
That is the clearest single example on the timeline of a workaround graduating into architecture,
and it supports prediction 4 in the app: the fix for softmax's "must sum to 1" problem is moving
into the normaliser.

---

## The three corrections to the session

Offered in the spirit of "if you catch me in another one, tell me".

1. **"Vaswani invented it in 2018 and 17."** v1 is **12 Jun 2017**
   ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)); NeurIPS publication is Dec 2017. There is
   no 2018 version.
2. **"DeltaNet, that paper came out in 2024."** The mechanism is **22 Feb 2021**
   ([arXiv:2102.11174](https://arxiv.org/abs/2102.11174), Schlag, Irie & Schmidhuber), building on
   Schmidhuber's 1992 fast-weight controllers. June 2024 is when it became *trainable in parallel* —
   a real and separate contribution, and arguably the reason anyone ships it. Both are on the
   timeline as distinct entries, which is finding 4 above.
3. **"DroPE / drop rope."** Published as **DroPE**, Sakana AI, **13 Dec 2025**
   ([arXiv:2512.12167](https://arxiv.org/abs/2512.12167)). Its precursor NoPE (May 2023) had already
   shown the underlying effect. The V4 result described in class — drop RoPE, take a loss spike,
   recover under annealing — is the same phenomenon, arrived at independently.

## And one date I could not pin down

**NTK-aware scaled RoPE** is the weakest date on the page and is flagged as such in the app rather
than given false precision. It is a Reddit post, so there is no authoritative timestamp. What is
verifiable: YaRN §3.2 credits `bloc97, 2023`, and a HuggingFace TGI issue quoting it verbatim was
opened **30 Jun 2023**, so the post is on or before that date. The community-cited day is 28 Jun
2023. **The month is solid; treat the day as approximate.**
