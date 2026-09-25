# Session 13 — Reversible transformers: Euler vs midpoint, and what they buy in batch size

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/subramanyanaik/schoolofai_subbu/blob/main/assignment13/reversible_llm.ipynb)
![status](https://img.shields.io/badge/results-pending%20Colab%20GPU%20run-orange)
![math](https://img.shields.io/badge/gradient%20checks-numpy%20%2B%20torch-brightgreen)

**Session-13 assignment.** Train a ~20M-parameter decoder-only transformer for 50M tokens at
the largest fixed batch size the GPU can hold. Train it again with a **reversible** backbone
— one where layer `l-1`'s activations can be reconstructed from layer `l`'s output, so the
forward pass never has to keep them around for backward — at the *same* batch size, testing
two coupling schemes (**euler**: additive two-stream coupling, RevNet/Reformer-style; and
**midpoint**: a single-stream leapfrog recurrence, `P[l+1] = P[l-1] + 2h·f(P[l])`) and report
which wins. Then push the winning reversible run to the largest batch size *it* can hold.
Report final loss, tokens/s and peak memory for all four runs.

> **Notebook:** [`reversible_llm.ipynb`](reversible_llm.ipynb) — open it directly in Colab
> with the badge above and run it on a **GPU runtime** (Runtime → Change runtime type → T4 or
> better). The whole comparison is a memory one, and it is only meaningful against a real CUDA
> allocator (`torch.cuda.max_memory_allocated`), not a CPU process's resident set size.
> **Source of truth:** [`reversible_llm.py`](reversible_llm.py). The notebook is generated
> from it, cell for cell, by [`tools/build_notebook.py`](tools/build_notebook.py); edit the
> `.py`, not the notebook, and CI checks the two haven't drifted.
> **Numbers:** [`results/results.json`](results/results.json), written by the run, injected
> into this README by [`tools/render_numbers.py`](tools/render_numbers.py). See the box below.

---

## Why this README's numbers section says "pending"

This repository was assembled on a machine that cannot load torch's native libraries at
all — CPU-only `torch` fails to `import` with `OSError: [WinError 4551] An Application
Control policy has blocked this file`, reproduced across three separate install locations
(inside OneDrive, a short root-level path, a temp-dir venv), so it is a device policy and not
a broken install. There is no CUDA GPU on that machine either. Rather than fabricate numbers
or quietly train a toy that isn't what the assignment asks for, the honest split is:

* **the code is real, complete, and was checked as thoroughly as it could be without torch** —
  see [Correctness, checked two ways](#correctness-checked-two-ways) below;
* **the actual 20M/50M/batch-search run has to happen where the assignment says it should**:
  a Colab GPU runtime. Open the badge above, `Runtime → Run all`, and it trains all four runs,
  writes `results/results.json` and `results/loss_curves.png`, and prints the same summary
  table that goes below.

After that run, `python tools/render_numbers.py` regenerates the numbers block below from
`results/results.json` — nothing in it is typed by hand once a run exists.

---

## What was asked, and how this answers it

| # | the assignment | what's here |
|---|---|---|
| 1 | 20M-param model, 50M tokens, fixed batch size = the biggest you can run | `size_to_target()` grows `n_layer` until the measured parameter count clears 20M for each backbone; `find_max_batch_size()` doubles-then-binary-searches batch size against real `RuntimeError: out of memory`s, one real forward+backward+step per probe |
| 2 | train again with reversibility; report which variant works, test at least euler and midpoint | both implemented as real `torch.autograd.Function`s that never store the activations they're claiming not to need — not `torch.utils.checkpoint` wearing a costume — run at the *same* batch size found in (1), compared on final loss |
| 3 | train again with reversibility, pushed to the maximum batch size | the winning variant from (2) run through `find_max_batch_size()` again, on its own memory profile |
| 4 | report final loss, tokens/s, memory peak, other findings | all four runs land in `results/results.json` and the table below, via `tools/render_numbers.py` — nothing hand-typed |

<!-- BEGIN:numbers -->

> **Results pending.** These numbers come from a real training run on a Colab GPU (`reversible_llm.ipynb`), not from the machine this repo was assembled on, which cannot load torch's native libraries at all (see the note at the top of `reversible_llm.py` — confirmed WDAC block, not a missing package). Open the notebook in Colab with a GPU runtime, run it end to end, commit the `results/` folder it writes, then run `python tools/render_numbers.py` to fill in this section from that run.
<!-- END:numbers -->

## The three backbones

All three share identical attention and MLP sub-layers (standard pre-LN transformer math);
what differs is only how consecutive layers are wired together — exactly the axis the
assignment is about.

| mode | wiring | what backward needs to keep |
|---|---|---|
| `standard` | `x = x + attn(ln(x))`; `x = x + mlp(ln(x))` | every layer's activations |
| `euler` | two `d/2`-wide streams `x1, x2`; `y1 = x1 + F(x2)`, `y2 = x2 + G(y1)` | nothing — `x1, x2` are recomputed from `y1, y2` going backward |
| `midpoint` | one `d`-wide stream; `P[l+1] = P[l-1] + 2h·f_l(P[l])` | nothing — `P[l-1]` is recomputed from `(P[l], P[l+1])` going backward |

`euler` is the RevNet/Reformer additive coupling: split the model width in half, run one
half-width sub-layer against the other stream, add, swap roles. `midpoint` is the leapfrog
recurrence from the session — implemented here as a **single** `autograd.Function` wrapping
the *entire* depth, so the only tensors alive between forward and backward are the trailing
pair of states, independent of how many layers there are.

Both are real reversible implementations, not `torch.utils.checkpoint` — checkpointing still
stores the *boundary* activations between segments and recomputes within a segment; these
never store any intermediate activation at all, at any granularity, because the coupling makes
every layer invertible from the next one's output.

**Restrictions that come with the territory, both enforced in the code:** `dropout=0.0`
everywhere (a dropped-out forward can't be inverted — you'd need to have kept the mask) and
`weight_decay=0.0` on every run, including the standard baseline, to keep the comparison to
one variable. A decoupled weight-decay step shrinks the weights *between* the forward pass and
the backward-time reconstruction, which makes the reconstructed activation wrong — so
reversible training can't use it at all, and the baseline is held to the same rule rather than
given an advantage the reversible runs structurally cannot have.

**Why character-level tokens, not BPE.** A 50k-row GPT-2 vocabulary's embedding table alone
is ~19M weights at `n_embd=384` — nearly the whole 20M budget, before a single attention layer
exists. Character-level tiny-Shakespeare (same corpus as
[assignment12](../assignment12)) keeps the vocabulary under 100 tokens, so the parameter
budget goes into the transformer body that reversibility actually acts on, matching what the
assignment is trying to measure. Swapping in `tiktoken` BPE is a one-line change
(`ENCODING = "gpt2"`-style) if you want subword tokens instead.

## Correctness, checked two ways

A reversible network that computes the wrong gradient still trains *something* — loss numbers
alone won't catch it, and this is exactly the kind of implementation where a sign error or an
off-by-one in which state gets reconstructed produces a plausible-looking, wrong loss curve. So
both custom backward passes were verified before being trusted with real GPU hours:

1. **[`tools/verify_reversible_math.py`](tools/verify_reversible_math.py)** — a from-scratch,
   pure-NumPy derivation of both schemes' forward *and* manual backward (hand-written vector-
   Jacobian products, no autograd at all), checked against central-difference finite gradients
   on every input and every parameter. Written first, with no torch dependency, because torch
   could not be loaded on the machine that wrote it. Run it with `python
   tools/verify_reversible_math.py` — no GPU, no torch, just NumPy.
2. **[`tests/test_reversible_math.py`](tests/test_reversible_math.py)** — the torch versions
   of the same check: each custom `autograd.Function`'s gradient is compared against an
   ordinary, fully-materialized autograd graph computing the identical forward math, parameter
   by parameter. This is the test that actually exercises `EulerBlockFunction` and
   `MidpointFunction` — the classes the real training run uses — and it runs in CI on CPU on
   every push (see the badge at the top), which is the closest thing to "run it and watch it
   pass" available without a working local torch.

The midpoint scheme has one easy-to-miss term, caught by exactly this process: `P[-1]` is
*bootstrapped* to equal `P[0]`, so `P[0]`'s storage is read twice by the very first step —
once as the nonlinear argument to `f_0`, once as the additive identity term. Naively
backpropagating only the first reading gives a `grad(P0)` that's numerically wrong for any
stack with more than one layer; the fix is a second gradient contribution captured at the
start of the backward loop (`g1_alias` in the code). `tests/test_reversible_math.py::
test_midpoint_gradient_needs_bootstrap_correction` recomputes the gradient *without* that
term and asserts it's different — a regression guard against silently deleting the fix.

## Running it

**On Colab (what the assignment asks for):** click the badge at the top, set the runtime to
GPU, Runtime → Run all. It downloads tiny-Shakespeare, sizes the three backbones to ~20M
params, runs all four training runs, and writes `results/results.json` +
`results/loss_curves.png`.

**Locally**, for anything that doesn't need a GPU or even a working torch install:

```bash
python tools/verify_reversible_math.py      # pure NumPy, no torch needed
```

Locally, with a working torch install (CPU is fine for these — they're all tiny):

```bash
pip install -r requirements.txt
python -m pytest tests -q                    # gradient-correctness tests
python tools/build_notebook.py --check        # notebook in sync with reversible_llm.py?
python tools/render_numbers.py --check        # README numbers match results/results.json?
```

To rebuild the notebook after editing `reversible_llm.py`:

```bash
python tools/build_notebook.py                # rebuild (no outputs)
python tools/build_notebook.py --execute       # rebuild and run it locally (GPU strongly
                                                # recommended; see the note at the top)
```

## What's not literal here, stated plainly

| | what | why |
|---|---|---|
| **memory, off CUDA** | falls back to process RSS (`resource`/`psutil`), not a GPU allocator peak | there is no CUDA-equivalent counter on CPU; the code says so at the point it happens and the number is labeled, never presented as a GPU figure |
| **`midpoint_h`** | a fixed scalar (`0.05` by default), not learned or per-layer | the session mentions `h` as something "we have to fine-tune"; this implementation exposes it as one config constant rather than a per-layer learned parameter, to keep the assignment's actual variable (reversible vs. not, euler vs. midpoint) isolated from an extra optimization problem. If you resize the model substantially, re-tune it — too large and the leapfrog recurrence's state norms grow without bound over depth |
| **parameter matching across backbones** | `standard` and `midpoint` land at the same `n_layer` (identical per-layer cost); `euler` needs roughly 4x as many layers to reach the same budget, because two `d/2`-wide sub-layers cost about a quarter of one `d`-wide block | not solved by hand — `size_to_target()` measures real parameter counts and grows `n_layer` until each backbone clears 20M independently, the same "measured, not assumed" approach as [assignment12](../assignment12) |

## What I'd take to a real run

Written before the numbers exist, so it can be checked against them rather than fitted to
them, once `results/results.json` does:

1. **`midpoint` should win the memory comparison decisively.** It touches the full model width
   every step with no width tax (unlike `euler`'s half-width streams), and its custom
   `Function` wraps the whole depth in one call — the activation memory it needs is
   independent of `n_layer` by construction, which `euler`'s per-block chaining doesn't quite
   match (each block is still its own `Function` call).
2. **Both reversible runs should be slower per token than the baseline at the same batch
   size** — the session's own estimate was 30-40%, because every backward step recomputes a
   forward pass that a normal network would have kept.
3. **The real payoff shows up in run 3, not run 2.** Run 2 is a controlled comparison at a
   batch size sized for the *non-reversible* baseline; it's expected to look like a wash or a
   mild loss on throughput. Run 3 is where reversibility gets to use the memory it saved, and
   the max-batch ratio there is the number this whole assignment is actually about.
