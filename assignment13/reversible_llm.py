# %% [markdown]
# # Session 13 — Reversible transformers: Euler vs midpoint, and what they buy in batch size
#
# **Assignment.** Train a ~20M-parameter decoder-only transformer for 50M tokens on a fixed,
# "biggest batch size you can run" batch. Then train it again with a *reversible* backbone —
# a network where layer `l-1`'s activations can be reconstructed from layer `l`'s output, so
# the forward pass never has to keep them around for backward — and test two coupling
# schemes: **euler** (additive two-stream coupling, RevNet/Reformer-style) and **midpoint**
# (a single-stream leapfrog recurrence, `P[l+1] = P[l-1] + 2h*f(P[l])`). Report which variant
# wins, then push the winning reversible run to the largest batch size it can hold, and report
# final loss, tokens/s and peak memory for all four runs.
#
# **Source of truth:** this file. [`reversible_llm.ipynb`](reversible_llm.ipynb) is generated
# from it cell-for-cell by [`tools/build_notebook.py`](tools/build_notebook.py); edit the
# `.py`, not the notebook.
#
# **Run this on a Colab GPU runtime** — Runtime → Change runtime type → GPU. The whole
# point of the assignment is a memory comparison, and that comparison is only meaningful
# against a real CUDA allocator (`torch.cuda.max_memory_allocated`), not a CPU process's
# resident set size. The notebook falls back to CPU and still runs correctly if no GPU is
# available, but the memory numbers it reports in that case are not comparable to the ones
# in the write-up.

# %%
import json
import os
import time
import urllib.request
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  no CUDA GPU visible — running on CPU/MPS. Memory numbers below will be process "
          "RSS, not GPU allocator peaks, and are not the numbers the assignment asks for. "
          "Open this notebook in Colab with a GPU runtime for the real run.")

RESULTS = {"device": DEVICE, "seed": SEED, "runs": {}}

# %% [markdown]
# ## 1. Data — character-level tiny-Shakespeare
#
# The model's parameter budget is the whole point of this assignment (does reversibility let
# a fixed GPU hold a bigger batch), so the vocabulary is kept deliberately small: a
# byte-pair tokenizer's 50k-row embedding table would eat most of a "20M-parameter" budget by
# itself and leave almost nothing for the transformer body that reversibility actually acts
# on. Character-level Shakespeare (as in [assignment12](../assignment12)) keeps ~99% of the
# parameter count inside the attention/MLP stack, where it belongs. Swapping in a `tiktoken`
# BPE vocabulary is a one-line change (`ENCODING = "gpt2"` below) if you want subword tokens
# instead — the rest of the pipeline does not care.

# %%
CORPUS = os.path.join(OUT_DIR, "tinyshakespeare.txt")
if not os.path.exists(CORPUS):
    try:
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
               "tinyshakespeare/input.txt")
        urllib.request.urlretrieve(url, CORPUS)
    except Exception as e:
        print(f"download failed ({e}); falling back to a synthetic corpus")
        rng = np.random.default_rng(SEED)
        words = ["thou", "art", "the", "king", "and", "night", "doth", "come", "sweet",
                 "sorrow", "my", "lord", "speak", "hence", "away", "shall", "heart"]
        text = " ".join(rng.choice(words, 400000))
        open(CORPUS, "w", encoding="utf-8").write(text)
text = open(CORPUS, encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(chars)
all_tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_val = max(1, int(0.05 * len(all_tokens)))
train_tokens, val_tokens = all_tokens[:-n_val], all_tokens[-n_val:]
print(f"corpus: {len(text):,} characters, {VOCAB_SIZE} distinct tokens, "
      f"{len(train_tokens):,} train / {len(val_tokens):,} val")


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# %% [markdown]
# ## 2. The three backbones
#
# All three share the same attention and MLP sub-layers (standard pre-LN transformer math);
# what differs is only how consecutive layers are *wired together*, which is exactly the
# axis the assignment is about:
#
# | mode | wiring | what backward needs to keep |
# |---|---|---|
# | `standard` | `x = x + attn(ln(x))`, `x = x + mlp(ln(x))` | every layer's activations |
# | `euler` | two streams `x1,x2`; `y1 = x1 + F(x2)`, `y2 = x2 + G(y1)` | nothing — `x1,x2` are recomputed from `y1,y2` on the way back |
# | `midpoint` | one stream; `P[l+1] = P[l-1] + 2h*f(P[l])` | nothing — `P[l-1]` is recomputed from `(P[l],P[l+1])` on the way back |
#
# `euler` is the classic RevNet/Reformer additive coupling, implemented block-by-block.
# `midpoint` is the leapfrog recurrence from the session, implemented as a single
# `autograd.Function` wrapping the *whole* stack, so the only tensors kept alive between
# forward and backward are the last two states — independent of depth. Both custom
# backward passes were derived and checked against finite differences in pure NumPy before
# being written here in torch (see [`tools/verify_reversible_math.py`](tools/verify_reversible_math.py)
# and [`tests/test_reversible_math.py`](tests/test_reversible_math.py), which runs the torch
# version of the same check in CI on every push).

# %%
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 8
    n_embd: int = 448
    n_head: int = 8
    dropout: float = 0.0  # 0 everywhere, on purpose -- see the note in section 3
    mode: str = "standard"  # "standard" | "euler" | "midpoint"
    midpoint_h: float = 0.05


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention over a `dim`-wide stream."""

    def __init__(self, dim, n_head, block_size, dropout):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.block_size = block_size

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc = nn.Linear(dim, 4 * dim, bias=False)
        self.proj = nn.Linear(4 * dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class SubBlock(nn.Module):
    """LayerNorm -> (attn|mlp), the unit both the standard and reversible backbones reuse."""

    def __init__(self, dim, n_head, block_size, dropout, kind):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.inner = (CausalSelfAttention(dim, n_head, block_size, dropout) if kind == "attn"
                      else MLP(dim, dropout))

    def forward(self, x):
        return self.inner(self.ln(x))


class FullBlock(nn.Module):
    """One attn+mlp transformer block: used by `standard` residuals and as the per-step
    function f_l inside the `midpoint` leapfrog recurrence (same math, different wiring)."""

    def __init__(self, dim, n_head, block_size, dropout):
        super().__init__()
        self.attn = SubBlock(dim, n_head, block_size, dropout, "attn")
        self.mlp = SubBlock(dim, n_head, block_size, dropout, "mlp")

    def forward(self, x):
        return self.attn(x) + self.mlp(x)


# %% [markdown]
# ### 2a. `standard` — plain pre-LN residual stack (the baseline)

# %%
class StandardStack(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            FullBlock(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
            for _ in range(cfg.n_layer)])

    def forward(self, x):
        for b in self.blocks:
            x = x + b.attn(x)
            x = x + b.mlp(x)
        return x


# %% [markdown]
# ### 2b. `euler` — additive two-stream coupling
#
# `y1 = x1 + F(x2)`, `y2 = x2 + G(y1)`. Invert with `x2 = y2 - G(y1)`, `x1 = y1 - F(x2)`.
# The custom `Function` keeps **only `y1, y2`** (which are needed as the next block's input
# anyway); `x1, x2` are reconstructed from them at backward time, `F` and `G` are re-run
# under `torch.enable_grad()` for exactly one block's worth of compute, and
# `torch.autograd.backward` deposits gradients straight onto `F`'s and `G`'s own
# parameters as a side effect — the classic RevNet/Reformer trick.

# %%
class EulerBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, f_block, g_block):
        ctx.f_block, ctx.g_block = f_block, g_block
        with torch.no_grad():
            y1 = x1 + f_block(x2)
            y2 = x2 + g_block(y1)
        ctx.save_for_backward(y1, y2)
        return y1, y2

    @staticmethod
    def backward(ctx, dy1, dy2):
        y1, y2 = ctx.saved_tensors
        f_block, g_block = ctx.f_block, ctx.g_block

        y1 = y1.detach().requires_grad_(True)
        with torch.enable_grad():
            gy1 = g_block(y1)
        torch.autograd.backward(gy1, dy2)  # accumulates onto g_block's own parameters
        with torch.no_grad():
            x2 = y2 - gy1.detach()
        dx1 = dy1 + y1.grad

        x2 = x2.detach().requires_grad_(True)
        with torch.enable_grad():
            fx2 = f_block(x2)
        torch.autograd.backward(fx2, dx1)  # accumulates onto f_block's own parameters
        dx2 = dy2 + x2.grad

        return dx1, dx2, None, None


class EulerCoupling(nn.Module):
    def __init__(self, dim_half, n_head, block_size, dropout):
        super().__init__()
        self.f = SubBlock(dim_half, n_head, block_size, dropout, "attn")
        self.g = SubBlock(dim_half, n_head, block_size, dropout, "mlp")

    def forward(self, x1, x2):
        return EulerBlockFunction.apply(x1, x2, self.f, self.g)


class EulerStack(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % 2 == 0
        half = cfg.n_embd // 2
        self.blocks = nn.ModuleList([
            EulerCoupling(half, cfg.n_head, cfg.block_size, cfg.dropout)
            for _ in range(cfg.n_layer)])

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        for blk in self.blocks:
            x1, x2 = blk(x1, x2)
        return torch.cat([x1, x2], dim=-1)


# %% [markdown]
# ### 2c. `midpoint` — single-stream leapfrog recurrence
#
# `P[l+1] = P[l-1] + 2h*f_l(P[l])`, bootstrapped with `P[-1] := P[0]`. Invert with
# `P[l-1] = P[l+1] - 2h*f_l(P[l])`. One `autograd.Function` wraps the *entire* depth: the
# forward loop runs under `no_grad`, keeping only the trailing pair `(P[L-1], P[L])` alive;
# the backward walks that pair back to `P[0]`, at each step reconstructing the previous state,
# recomputing `f_l` under `enable_grad` for the vector-Jacobian product, and — because every
# `P[l]` (for `0 < l < L`) feeds the recurrence *twice* (once through `f_l`, once as the
# additive carry two steps later) — accumulating gradient from **both** paths before moving
# on. `P[0]` gets one further correction for the `P[-1]:=P[0]` bootstrap, which reads `P[0]`'s
# storage a second time on the very first step; the derivation and its finite-difference check
# are in `tests/test_reversible_math.py`.

# %%
class MidpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P0, h, layers):
        ctx.h, ctx.layers = h, layers
        with torch.no_grad():
            Pprev, Pcur = P0, P0
            for layer in layers:
                Pnext = Pprev + 2 * h * layer(Pcur)
                Pprev, Pcur = Pcur, Pnext
        ctx.save_for_backward(Pprev, Pcur)  # P[L-1], P[L]
        return Pcur

    @staticmethod
    def backward(ctx, grad_PL):
        h, layers = ctx.h, ctx.layers
        L = len(layers)
        cur, nxt = ctx.saved_tensors  # P[L-1], P[L]  (cur plays "P[l]" starting at l=L-1)

        g_next = grad_PL          # g_{l+1}, seeded as g_L
        g_next2 = torch.zeros_like(grad_PL)  # g_{l+2}
        g1_alias = None

        for l in range(L - 1, -1, -1):
            if l == 0:
                g1_alias = g_next  # dL/dP1, needed for the P[-1]:=P[0] bootstrap correction
            cur_ = cur.detach().requires_grad_(True)
            with torch.enable_grad():
                f_out = layers[l](cur_)
            torch.autograd.backward(f_out, 2 * h * g_next)  # accumulates onto layers[l] params
            g_l = cur_.grad + g_next2
            with torch.no_grad():
                prev = nxt - 2 * h * f_out.detach()  # reconstruct P[l-1]
            nxt, cur = cur, prev
            g_next2, g_next = g_next, g_l

        grad_P0 = g_next + g1_alias
        return grad_P0, None, None


class MidpointStack(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.h = cfg.midpoint_h
        self.layers = nn.ModuleList([
            FullBlock(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
            for _ in range(cfg.n_layer)])

    def forward(self, x):
        return MidpointFunction.apply(x, self.h, list(self.layers))


# %% [markdown]
# ### 2d. Putting a backbone behind an embedding and a tied LM head

# %%
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.mode == "standard":
            self.backbone = StandardStack(cfg)
        elif cfg.mode == "euler":
            self.backbone = EulerStack(cfg)
        elif cfg.mode == "midpoint":
            self.backbone = MidpointStack(cfg)
        else:
            raise ValueError(cfg.mode)
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        x = self.backbone(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self):
        # tok_emb.weight and head.weight are the *same* nn.Parameter object (tied), and
        # nn.Module.parameters() already de-duplicates by identity, so this counts it once.
        return sum(p.numel() for p in self.parameters())


# %% [markdown]
# ## 3. Sizing all three backbones to ~20M parameters
#
# `standard` and `midpoint` reuse the identical `FullBlock` (full-width attn+mlp), so the
# same `n_layer` gives them the same parameter count. `euler` runs two `dim/2`-wide streams,
# so an `F`+`G` pair costs roughly a quarter of a full-width block (`(d/2)^2` vs `d^2`) and
# needs proportionally more layers to reach the same budget — which is also exactly why its
# activation-count-per-layer story is different and worth reporting on its own, not forced to
# match `n_layer` with the other two. Rather than solve that algebra by hand, this grows
# `n_layer` for each mode until the *measured* parameter count clears the target — the same
# "measured, not assumed" discipline as [assignment12](../assignment12).
#
# **No weight decay, anywhere, in any run below.** The reversible identity `P[l-1] =
# P[l+1] - 2h*f(P[l])` (and the euler analogue) is only exact if forward and backward see the
# *same* weights; a decoupled weight-decay step that shrinks the weights between the forward
# pass and the backward reconstruction would make the reconstructed activations wrong. So
# reversible training cannot use weight decay. To keep the comparison to one variable
# (reversibility itself, not regularization), `standard` is trained with `weight_decay=0` too
# — a real deployment would normally give the baseline nonzero decay, and that asymmetry is
# worth knowing about when reading the loss numbers below.

# %%
N_EMBD = 448
N_HEAD = 8
BLOCK_SIZE = 256
MIDPOINT_H = 0.05  # leapfrog step size; see the note in section 2c if you resize the model
TARGET_PARAMS = 20_000_000
TARGET_TOKENS = 50_000_000


def make_cfg(mode, n_layer):
    return GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=n_layer,
                      n_embd=N_EMBD, n_head=N_HEAD, dropout=0.0, mode=mode,
                      midpoint_h=MIDPOINT_H)


def size_to_target(mode, target=TARGET_PARAMS, max_layer=128):
    n_layer = 1
    while n_layer <= max_layer:
        cfg = make_cfg(mode, n_layer)
        params = GPT(cfg).num_params()
        if params >= target:
            return cfg, params
        n_layer += 1
    raise RuntimeError(f"could not reach {target:,} params for mode={mode} "
                        f"within {max_layer} layers")


CONFIGS, RESULTS["model_sizing"] = {}, {}
for _mode in ("standard", "euler", "midpoint"):
    _cfg, _params = size_to_target(_mode)
    CONFIGS[_mode] = _cfg
    RESULTS["model_sizing"][_mode] = {
        "n_layer": _cfg.n_layer, "n_embd": _cfg.n_embd, "n_head": _cfg.n_head,
        "params": _params}
    print(f"{_mode:9s}: n_layer={_cfg.n_layer:3d}  n_embd={_cfg.n_embd}  -> "
          f"{_params:,} params ({_params / TARGET_PARAMS - 1:+.1%} vs {TARGET_PARAMS:,} target)")

# %% [markdown]
# ## 4. Memory accounting and the batch-size search
#
# On a CUDA device, "peak memory" is `torch.cuda.max_memory_allocated()` between a
# `reset_peak_memory_stats()` and the measurement point — the real allocator high-water mark,
# not an estimate. Off CUDA there is no such counter, so this falls back to the process's
# resident set size, which is a much cruder number (it includes the Python interpreter, the
# tokenizer, everything) and is reported as such rather than dressed up as a GPU figure.
#
# "The biggest batch size you can run" is answered the same way for every run below:
# double the batch size until a step raises `RuntimeError: ... out of memory`, then binary
# search the gap to within ~10%. Each probe runs one real forward+backward+optimizer step —
# an OOM caught before the step completes proves nothing about what backward would have
# needed.

# %%
def reset_peak_memory():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss * (1 if os.uname().sysname == "Darwin" else 1024)  # macOS: bytes, Linux: KiB
    except Exception:
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            return -1


def _oom(e):
    return "out of memory" in str(e).lower()


def try_step(model, optimizer, cfg, batch_size):
    x, y = get_batch(train_tokens, batch_size, cfg.block_size, DEVICE)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return float(loss.item())


def find_max_batch_size(mode, cfg, start=4, cap=8192):
    model = GPT(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

    def probe(bs):
        reset_peak_memory()
        try_step(model, optimizer, cfg, bs)

    bs, last_ok = start, 0
    while bs <= cap:
        try:
            probe(bs)
            last_ok, bs = bs, bs * 2
        except RuntimeError as e:
            if not _oom(e):
                raise
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            break
    if last_ok == 0:
        raise RuntimeError(f"[{mode}] even batch size {start} does not fit in memory")
    lo, hi = last_ok, min(bs, cap)
    while hi - lo > max(1, lo // 10):
        mid = (lo + hi) // 2
        try:
            probe(mid)
            lo = mid
        except RuntimeError as e:
            if not _oom(e):
                raise
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            hi = mid
    del model, optimizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return lo


# %% [markdown]
# ## 5. The training loop
#
# One function, reused for all four runs, so the only things that differ between them are the
# arguments: `mode` (which backbone), `batch_size`, and how many steps it takes to reach
# `target_tokens` at that batch size. Every run trains for the *same number of tokens*
# (50M by default) — batch size only changes how many gradient steps that is.

# %%
def train_run(name, mode, cfg, batch_size, target_tokens, lr=3e-4, weight_decay=0.0,
              log_every=50):
    torch.manual_seed(SEED)
    model = GPT(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_steps = max(1, target_tokens // (batch_size * cfg.block_size))

    reset_peak_memory()
    losses = []
    tokens_seen = 0
    t0 = time.time()
    for step in range(n_steps):
        x, y = get_batch(train_tokens, batch_size, cfg.block_size, DEVICE)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        tokens_seen += batch_size * cfg.block_size
        losses.append(float(loss.item()))
        if step % log_every == 0 or step == n_steps - 1:
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            print(f"[{name}] step {step:5d}/{n_steps}  loss {loss.item():.4f}  "
                  f"{tokens_seen / max(elapsed, 1e-9):,.0f} tok/s  "
                  f"{tokens_seen:,}/{target_tokens:,} tokens")

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    peak_mem = peak_memory_bytes()
    thin = max(1, len(losses) // 200)
    result = {
        "mode": mode, "batch_size": batch_size, "block_size": cfg.block_size,
        "n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "params": model.num_params(),
        "steps": n_steps, "target_tokens": target_tokens, "tokens_trained": tokens_seen,
        "final_loss": losses[-1], "min_loss": min(losses), "loss_curve": losses[::thin],
        "wall_clock_s": wall, "tokens_per_s": tokens_seen / max(wall, 1e-9),
        "peak_memory_bytes": peak_mem,
        "peak_memory_mib": (peak_mem / (1024 ** 2)) if peak_mem >= 0 else None,
        "weight_decay": weight_decay, "lr": lr,
    }
    RESULTS["runs"][name] = result
    del model, optimizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result


# %% [markdown]
# ## 6. Run 1 — the standard baseline, at the biggest batch size it can hold
#
# This fixes the batch size that runs 2a/2b reuse, so the *only* thing that changes between
# "run 1" and "run 2" is whether the backbone is reversible.

# %%
print("finding the largest batch size the STANDARD baseline can run...")
BASELINE_BATCH = find_max_batch_size("standard", CONFIGS["standard"])
RESULTS["baseline_max_batch"] = BASELINE_BATCH
print(f"baseline max batch size: {BASELINE_BATCH}\n")

run_baseline = train_run("baseline_standard", "standard", CONFIGS["standard"],
                          batch_size=BASELINE_BATCH, target_tokens=TARGET_TOKENS,
                          weight_decay=0.0)

# %% [markdown]
# ## 7. Run 2 — reversible, same fixed batch size: euler vs. midpoint
#
# Same token budget, same batch size, same optimizer — the only variable is the coupling
# scheme. Winner is whichever reaches the lower final training loss; ties broken by
# throughput.

# %%
run_euler = train_run("reversible_euler_fixedbatch", "euler", CONFIGS["euler"],
                       batch_size=BASELINE_BATCH, target_tokens=TARGET_TOKENS,
                       weight_decay=0.0)
run_midpoint = train_run("reversible_midpoint_fixedbatch", "midpoint", CONFIGS["midpoint"],
                          batch_size=BASELINE_BATCH, target_tokens=TARGET_TOKENS,
                          weight_decay=0.0)

if run_euler["final_loss"] < run_midpoint["final_loss"]:
    WINNER = "euler"
elif run_midpoint["final_loss"] < run_euler["final_loss"]:
    WINNER = "midpoint"
else:
    WINNER = "euler" if run_euler["tokens_per_s"] >= run_midpoint["tokens_per_s"] else "midpoint"
RESULTS["winner_variant"] = WINNER
print(f"\nwinning reversible variant at fixed batch {BASELINE_BATCH}: {WINNER}  "
      f"(euler final loss {run_euler['final_loss']:.4f} vs. "
      f"midpoint final loss {run_midpoint['final_loss']:.4f})")

# %% [markdown]
# ## 8. Run 3 — the winning reversible variant, pushed to its own maximum batch size
#
# `standard` has to keep every layer's activations; `euler` and `midpoint` only ever hold the
# last one or two states. That is the whole bet of this assignment, and this is where it
# either pays off or doesn't: the same GPU, the same model family, and how much bigger a
# batch the reversible backbone can actually hold.

# %%
print(f"finding the largest batch size the {WINNER.upper()} reversible run can hold...")
WINNER_MAX_BATCH = find_max_batch_size(WINNER, CONFIGS[WINNER])
RESULTS[f"{WINNER}_max_batch"] = WINNER_MAX_BATCH
print(f"{WINNER} max batch size: {WINNER_MAX_BATCH}  "
      f"({WINNER_MAX_BATCH / BASELINE_BATCH:.1f}x the baseline's {BASELINE_BATCH})\n")

run_pushed = train_run(f"reversible_{WINNER}_maxbatch", WINNER, CONFIGS[WINNER],
                        batch_size=WINNER_MAX_BATCH, target_tokens=TARGET_TOKENS,
                        weight_decay=0.0)

# %% [markdown]
# ## 9. Results
#
# Everything printed above is also written to `results/results.json`; `README.md`'s numbers
# are rendered from that file by `tools/render_numbers.py` and are never typed in by hand.

# %%
with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(RESULTS, fh, indent=2)
print(f"wrote {os.path.join(OUT_DIR, 'results.json')}")

plt.figure(figsize=(8, 5))
for _name, _run in RESULTS["runs"].items():
    _ys = _run["loss_curve"]
    _xs = np.linspace(0, _run["tokens_trained"], len(_ys))
    plt.plot(_xs, _ys, label=f"{_name} (batch={_run['batch_size']})")
plt.xlabel("tokens trained")
plt.ylabel("training loss")
plt.legend()
plt.title("Session 13 - reversible transformer training runs")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=140)
plt.close()
print("wrote results/loss_curves.png")

print("\n| run | mode | batch | params | final loss | tok/s | peak mem |")
print("|---|---|---|---|---|---|---|")
for _name, _run in RESULTS["runs"].items():
    _mem = f"{_run['peak_memory_mib']:.0f} MiB" if _run["peak_memory_mib"] is not None else "n/a"
    print(f"| {_name} | {_run['mode']} | {_run['batch_size']} | {_run['params']:,} | "
          f"{_run['final_loss']:.4f} | {_run['tokens_per_s']:,.0f} | {_mem} |")
