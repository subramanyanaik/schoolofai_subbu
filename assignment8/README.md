# How Attention Works Now

[![validate](https://github.com/subramanyanaik/schoolofai_subbu/actions/workflows/validate.yml/badge.svg)](https://github.com/subramanyanaik/schoolofai_subbu/actions/workflows/validate.yml)
![mechanisms](https://img.shields.io/badge/mechanisms-39-blue)
![session](https://img.shields.io/badge/from%20Session%208-24-brightgreen)
![extras](https://img.shields.io/badge/added%20beyond%20the%20session-15-orange)
![dates](https://img.shields.io/badge/dates%20verified%20against%20primary%20sources-39%2F39-brightgreen)
![caveats](https://img.shields.io/badge/dates%20flagged%20as%20ambiguous-12-yellow)
![deps](https://img.shields.io/badge/dependencies-0-blue)

**Session-8 assignment.** A single-page web app that lays out every attention mechanism
**in the order it was actually launched**, and explains each one as an answer to the problem that
existed at that moment.

> **Live:** _(Netlify — see [Deploying](#deploying); publish directory `assignment8/site`)_
> **Repo:** https://github.com/subramanyanaik/schoolofai_subbu/tree/main/assignment8

---

## The argument the page is making

Vanilla attention was never *wrong*. It was **expensive**, in exactly two ways, and everything since
is somebody looking at one of those two bills and trying to pay less of it.

| | Bill 1 — compute | Bill 2 — memory |
|---|---|---|
| what it is | the N×N score matrix | the KV cache, per user, forever |
| how it grows | quadratic — 1K→1M tokens is a **million×** more work | linear — but it decides how many users you can serve |
| who attacks it | sparse patterns, selection, linear attention | MQA, GQA, MLA, sliding window, hybrids |

Ordering by **date** rather than by family is the whole point, because the ordering is what shows you
the field changing its mind:

| Era | Years | Thesis |
|---|---|---|
| 1. Exactness | 2014–2018 | Look at everything, and prove it helps. |
| 2. The bill arrives | 2019–2020 | N² is quadratic and nobody can pay it. |
| 3. Memory back, quietly | 2021–2022 | Approximation was too lossy. Get exactness back, cheaper. |
| 4. Length, at any cost | 2023–2024 | Ship a bigger number on the context box. |
| 5. Memory back, again | 2025–2026 | Stop choosing — put different attention in different layers. |

Exactness → length → memory → length → memory. Once you see that oscillation you can guess what
comes next, which is what the final section of the page does.

---

## What's in it

**Six things that actually compute, not six pictures of computations:**

| Lab | What it does |
|---|---|
| **Attention playground** | Real scaled-dot-product attention over a toy sentence. Toggle `÷√dₖ`, the causal mask and softmax, and drag the amplitude. Rows genuinely sum to 1; the repeated token really does get an identical key vector. |
| **Sparsity lab** | Nine patterns on the same 48×48 causal grid — dense, sliding window, sinks, strided, BigBird, top-k, NSA, DSA, CSA — with **two** cost bars: measured density on the grid, and projected keys-per-query at 128K using each paper's own constants. They disagree wildly, and that disagreement is the lesson. |
| **RoPE circle** | Two tokens rotating. The absolute angles move; the gap does not. |
| **Decay curves** | Score-vs-distance for RoPE, ALiBi and sliding window, against what retrieval actually needs. |
| **RoPE extension** | Wavelength per dimension under plain RoPE / PI / NTK-aware / YaRN, with the training-context line drawn in — you can see exactly which dimensions each method squashes. |
| **KV calculator** | The session's own arithmetic. MHA vs GQA vs MQA vs MLA vs SWA vs a 1-in-4 hybrid, live in GB. |
| **Linear state + delta rule** | The state stays d×d while the cache grows; and the instructor's own `40 + 55 = 95` vs `→ 55` example, playable. |

Every mechanism gets: the problem it answered, what it does, the math, **honest pros and cons**, and
*when you would actually pick it*. **155 documented downsides across 39 mechanisms** — there is no
entry on this page with only pros, because a technique written up with only pros has not been
understood yet.

---

## Dates — the part that is easiest to get wrong

**Question 2 is answered in [ANSWERS.md](ANSWERS.md)** — what the date order showed, and the 15 mechanisms the session did not cover.

Full table with a source link for every single date: **[DATES.md](DATES.md)**, generated directly
from [`site/data.js`](site/data.js) so the README and the site cannot drift apart.

**Method.** Every date is the **v1 arXiv submission timestamp**, pulled from the arXiv API
(`export.arxiv.org/api/query?id_list=…`, the `<published>` field) and cross-checked against the
`[Submitted on …]` line on the paper's abstract page. Not the announcement date, not the conference
publication date, not the `v3` you happen to be reading. For things that were never papers, the
official release post or repository.

**Where the sources disagreed, I recorded the disagreement instead of picking one quietly.**
Twelve entries carry a caveat. The ones worth knowing about:

- **DeepSeek-V4 (CSA + HCA)** — the arXiv identifier is `2606.*`, which implies a June 2026
  announcement, but the v1 submission line reads **26 Apr 2026**, and the model preview shipped
  **24 Apr 2026**. One aggregator reports 19 Jun 2026, which is the announcement. I used the
  submission date and said so.
- **NTK-aware scaled RoPE** — this is the weakest date on the page and it is flagged as such. It is
  a **Reddit post**, not a paper, so there is no authoritative timestamp, and Reddit was not
  fetchable from my environment. What is verifiable: YaRN §3.2 credits `bloc97, 2023`, and a
  HuggingFace TGI issue quoting it verbatim was opened **30 Jun 2023**, so the post is on or before
  that. The community-cited day is 28 Jun 2023. **The month is solid; treat the day as approximate.**
- **YaRN** — identifier is `2309.*` but the v1 submission is **31 Aug 2023**. Submission date used.
- **DeltaNet** — the session dated it to 2024. The delta rule for linear attention is
  **Schlag, Irie & Schmidhuber, 22 Feb 2021**; what landed in June 2024 is the chunkwise *parallel
  training algorithm* that made it usable at scale. Both are on the timeline, separately.
- **Qwen3-Next** — the 3:1 linear:full layer ratio is widely quoted but I could not confirm it from
  a primary source, **so the page does not assert it**. Kimi Linear's 3:1 *is* stated in its own
  abstract and is cited there instead.

### Corrections to the session, offered in the spirit of "if you catch me in another one, tell me"

1. **"Vaswani invented it in 2018 and 17."** — v1 is **12 Jun 2017**; NeurIPS publication is
   Dec 2017. There is no 2018 version.
2. **"DeltaNet, that paper came out in 2024."** — the mechanism is **Feb 2021**
   ([arXiv:2102.11174](https://arxiv.org/abs/2102.11174)), building on Schmidhuber's 1992 fast-weight
   controllers. 2024 is when it became *trainable in parallel*, which is a different and also
   important claim.
3. **"DroPE / drop rope"** — published as **DroPE**, Sakana AI, **13 Dec 2025**
   ([arXiv:2512.12167](https://arxiv.org/abs/2512.12167)). Worth adding that its precursor **NoPE**
   ([arXiv:2305.19466](https://arxiv.org/abs/2305.19466), 31 May 2023) had already shown causal
   decoders can infer position from the mask alone — DroPE's contribution is getting there from a
   *pretrained* RoPE model cheaply. The V4 result reported in class (drop RoPE → loss spike →
   recover under annealing) is the same phenomenon, found independently.

---

## The 15 mechanisms added beyond the session

The session ran out of time before several things it would clearly have covered. Added, with the
same treatment:

**Position** — Bahdanau additive attention (2014), Luong dot-product (2015), Shaw relative
positions (2018), Transformer-XL (2019), Position Interpolation (2023), LongRoPE (2024).
**Sparse** — Reformer/LSH (2020), BigBird (2020), MoBA (2025).
**Linear** — Performer (2020), Infini-attention (2024), Kimi Delta Attention (2025).
**Exact / systems** — FlashAttention (2022), PagedAttention (2023), Differential Transformer (2024).

Two of these matter more than their obscurity suggests. **FlashAttention** is the reason most of the
2020 "efficient transformer" wave died — it made exact dense attention faster than the
approximations. And **PagedAttention** is the honest asterisk on a lot of 2024 "1M context" claims,
which were as much about allocation strategy as about attention math.

---

## Running it

Zero dependencies, zero build step, no external network requests. It is four static files.

```bash
python -m http.server 8791 --directory assignment8/site
```

Then open http://localhost:8791.

Regenerate the date table after editing `site/data.js`:

```bash
python assignment8/tools/gen_date_table.py
```

## Deploying

**Exactly five files get hosted** — everything in `assignment8/site/`:

```
index.html    styles.css    data.js    viz.js    app.js
```

Nothing else. `README.md`, `DATES.md` and `tools/` are repo documentation and do not belong on
the web root. There is no build step, no `node_modules`, no bundler output, and the page makes
zero external network requests.

**Netlify (repo-connected).** [`netlify.toml`](../netlify.toml) at the repo root already sets
`publish = "assignment8/site"`, so connecting the repo needs no dashboard configuration — leave
the build command empty and let the file do the work. It also sets a strict
`Content-Security-Policy` of `default-src 'none'`, which the site satisfies because every asset
is same-origin.

**Netlify (drag and drop).** Drop the `assignment8/site` folder itself onto
[app.netlify.com/drop](https://app.netlify.com/drop). Drop the *folder*, not the repo — otherwise
`index.html` will not be at the site root.

**Anywhere else.** Vercel, Cloudflare Pages and GitHub Pages all work the same way: serve
`assignment8/site` as the document root, no build command.

## Layout

```
assignment8/
├── README.md                 this file
├── DATES.md                  every date + source, generated from data.js
├── site/
│   ├── index.html            structure and prose
│   ├── styles.css            theme-aware, light and dark
│   ├── data.js               all 39 mechanisms — the single source of truth
│   ├── viz.js                the attention arithmetic + all canvas drawing
│   └── app.js                rendering and interaction wiring
└── tools/
    └── gen_date_table.py     regenerates DATES.md from data.js
```

`data.js` is deliberately the only place any claim lives. The timeline, the comparison table, the
sources appendix and `DATES.md` are all generated from it.

## Sources

Every date links to its primary source in [DATES.md](DATES.md) and in the app's Sources section
(92 source links). Content is grounded in the **ERA V5 Session 8 transcript of 15 Aug 2026** and
checked against the primary literature; where the two disagree, the paper wins and the disagreement
is written down.
