# %% [markdown]
# # Session 14 — Grow a dense model into a mixture of experts, then keep training it
#
# **Assignment.** Train a linear (dense, one-feed-forward-network-per-layer) decoder-only
# transformer. Convert it into a mixture-of-experts model. Show that training continues —
# and the loss keeps dropping — after the conversion. Model size and data are an open choice;
# this picks a character-level model small enough to train for real, twice, on a laptop GPU,
# rather than a size chosen to hit a parameter target.
#
# **Source of truth:** this file. [`moe_llm.ipynb`](moe_llm.ipynb) is generated from it
# cell-for-cell by [`tools/build_notebook.py`](tools/build_notebook.py); edit the `.py`, not
# the notebook.
#
# **What "convert to MoE" means here — sparse upcycling, copy method.** Section 14 of the
# course covers two ways to grow a dense feed-forward block into `E` experts: *partition* (cut
# the dense hidden layer into `E` slices) and *copy* (clone the whole dense feed-forward network
# `E` times, add a little noise, let the router and continued training tell them apart) — citing
# Komatsuzaki et al. 2022, "sparse upcycling". This file uses **copy**, because it has a clean,
# checkable invariant: with zero noise, the MoE layer computes the *exact* dense output no
# matter which experts a randomly initialized router happens to pick — see
# `test_zero_noise_upcycling_reconstructs_dense_exactly` in `tests/test_moe_math.py`. A small
# noise is added in the real run specifically to break that equivalence and give the router and
# continued training something to do; without it the model would silently stay a disguised
# dense net forever.
#
# **Routing.** Softmax over all `E` router logits, keep the top `k`, renormalize those `k`
# probabilities to sum to 1 — exactly as the session described Qwen3's router. No shared
# expert (Qwen3's own count is 0, per the session's model table).
#
# **Balancing.** Not an auxiliary loss added to the training objective — the session's own
# conclusion was that the 2024-era auxiliary-loss approach destabilizes because its gradient
# fights the language-modeling gradient through the same router weights. Instead: a per-expert
# bias, invisible to the gate's weighting, added only to the *selection* score, nudged down for
# experts that took more than their fair share of the last window of tokens and up for experts
# that took less — DeepSeek-V3's "auxiliary-loss-free" scheme. Also per the session: balance
# over a *window* of steps, not every micro-batch, so an expert gets a fair chance to catch up
# before being judged.

# %%
import json
import os
import time
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

RESULTS = {"device": DEVICE, "seed": SEED, "runs": {}}

# %% [markdown]
# ## 1. Data — character-level tiny-Shakespeare
#
# Same corpus and reasoning as [assignment12](../assignment12) and
# [assignment13](../assignment13): a byte-pair vocabulary's embedding table would dominate a
# small model's parameter count before a single transformer layer exists, which would make
# "total vs. active parameters" (section 14's own running theme) a statement about the
# embedding table rather than about the feed-forward blocks this assignment is actually
# about. Character-level keeps the vocabulary under 100 tokens.

# %%
import urllib.request

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
# ## 2. The model — one config, two feed-forward blocks
#
# Attention, embeddings, and the residual wiring are completely untouched by the MoE
# conversion. The *only* thing that changes is what sits inside each block where a single
# feed-forward network used to be — exactly the layer the session's whole discussion of
# experts was about.

# %%
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_embd: int = 192
    n_head: int = 6
    mlp_mult: int = 4
    dropout: float = 0.0
    mode: str = "dense"       # "dense" | "moe"
    n_experts: int = 8
    top_k: int = 2
    balance_gamma: float = 0.002


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class DenseMLP(nn.Module):
    """The 'linear model' feed-forward block: expand by mlp_mult, GELU, project back. This
    is both the session 14 baseline *and* the exact unit that gets cloned into each expert."""

    def __init__(self, dim, mult, dropout):
        super().__init__()
        self.fc = nn.Linear(dim, mult * dim, bias=False)
        self.proj = nn.Linear(mult * dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


# %% [markdown]
# ### 2a. The router and the mixture-of-experts block
#
# `probs = softmax(logits)` over all `E` experts. Selection uses `probs + bias` (the
# loss-free-balancing bias, detached from autograd — it only ever moves by a fixed step size,
# never by gradient); the gate weight actually multiplying each chosen expert's output is the
# *unbiased* `probs`, gathered at the chosen indices and renormalized to sum to 1. That split
# — bias decides *who*, softmax decides *how much* — is exactly what keeps the balancing
# mechanism from fighting the language-modeling loss through the same numbers.
#
# Expert compute here is dense (every expert runs on every token, then unselected ones are
# zeroed by the gate) rather than a real gathered/sparse dispatch. That makes the implementation
# easy to check against plain tensor ops — which the tests below do — at the cost of not
# measuring the wall-clock/FLOPs speedup a real sparse dispatch would give; see "What's not
# literal here" in the README for the active-vs-total parameter accounting that stands in for
# it.

# %%
class MoEMLP(nn.Module):
    def __init__(self, dim, mult, n_experts, top_k, dropout, balance_gamma):
        super().__init__()
        assert 0 < top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.balance_gamma = balance_gamma
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([DenseMLP(dim, mult, dropout) for _ in range(n_experts)])
        self.register_buffer("bias", torch.zeros(n_experts))
        self.register_buffer("count_accum", torch.zeros(n_experts))

    def forward(self, x):
        B, T, C = x.shape
        logits = self.router(x)                       # (B,T,E) raw router logits
        probs = F.softmax(logits, dim=-1)              # "how much", never touched by bias
        score = probs + self.bias                      # "who" -- bias only affects selection
        _, top_idx = score.topk(self.top_k, dim=-1)     # (B,T,k)
        gate = probs.gather(-1, top_idx)
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        expert_out = torch.stack([e(x) for e in self.experts], dim=-2)  # (B,T,E,C)
        gate_full = torch.zeros(B, T, self.n_experts, device=x.device, dtype=gate.dtype)
        gate_full.scatter_(-1, top_idx, gate)
        out = (expert_out * gate_full.unsqueeze(-1)).sum(dim=-2)

        with torch.no_grad():
            onehot = torch.zeros(B, T, self.n_experts, device=x.device)
            onehot.scatter_(-1, top_idx, 1.0)
            self.count_accum += onehot.sum(dim=(0, 1))
        return out

    def maybe_rebalance(self):
        """Loss-free balancing update. Called by the caller on its own window schedule (not
        every forward call) -- see train_run's `balance_every`."""
        with torch.no_grad():
            total = self.count_accum.sum()
            if total <= 0:
                return None
            target = total / self.n_experts
            counts = self.count_accum.clone()
            self.bias[counts > target] -= self.balance_gamma
            self.bias[counts < target] += self.balance_gamma
            self.count_accum.zero_()
        return counts


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg.n_embd, cfg.n_head, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        if cfg.mode == "dense":
            self.mlp = DenseMLP(cfg.n_embd, cfg.mlp_mult, cfg.dropout)
        elif cfg.mode == "moe":
            self.mlp = MoEMLP(cfg.n_embd, cfg.mlp_mult, cfg.n_experts, cfg.top_k, cfg.dropout,
                               cfg.balance_gamma)
        else:
            raise ValueError(cfg.mode)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
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
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def param_counts(self):
        """(total, active) parameter counts. Equal for a dense model; for a MoE model,
        active subtracts the (n_experts - top_k) experts that never run for any given token,
        per layer -- the "active vs. total" split section 14 spent most of its time on."""
        total = self.num_params()
        if self.cfg.mode != "moe":
            return total, total
        inactive = 0
        for b in self.blocks:
            per_expert = sum(p.numel() for p in b.mlp.experts[0].parameters())
            inactive += per_expert * (b.mlp.n_experts - b.mlp.top_k)
        return total, total - inactive


# %% [markdown]
# ## 3. Sparse upcycling: dense -> MoE
#
# Attention, embeddings, both LayerNorms, and the LM head are copied verbatim -- the
# conversion never touches them. Each block's single `DenseMLP` is cloned `n_experts` times
# into the new block's expert list, with independent Gaussian noise added to each clone so
# they are not permanently identical (the router is freshly initialized either way). Setting
# `noise_std=0` reproduces the dense model's output exactly, by construction -- see
# `test_zero_noise_upcycling_reconstructs_dense_exactly`.

# %%
def upcycle_to_moe(dense_model: GPT, moe_cfg: GPTConfig, noise_std: float, seed: int = 0) -> GPT:
    assert dense_model.cfg.mode == "dense" and moe_cfg.mode == "moe"
    device = next(dense_model.parameters()).device
    moe_model = GPT(moe_cfg).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    moe_model.tok_emb.weight.data.copy_(dense_model.tok_emb.weight.data)
    moe_model.pos_emb.weight.data.copy_(dense_model.pos_emb.weight.data)
    moe_model.ln_f.load_state_dict(dense_model.ln_f.state_dict())

    for moe_block, dense_block in zip(moe_model.blocks, dense_model.blocks):
        moe_block.ln1.load_state_dict(dense_block.ln1.state_dict())
        moe_block.ln2.load_state_dict(dense_block.ln2.state_dict())
        moe_block.attn.load_state_dict(dense_block.attn.state_dict())
        for expert in moe_block.mlp.experts:
            expert.fc.weight.data.copy_(dense_block.mlp.fc.weight.data)
            expert.proj.weight.data.copy_(dense_block.mlp.proj.weight.data)
            if noise_std > 0:
                expert.fc.weight.data.add_(
                    torch.randn(expert.fc.weight.shape, generator=gen).to(device) * noise_std)
                expert.proj.weight.data.add_(
                    torch.randn(expert.proj.weight.shape, generator=gen).to(device) * noise_std)
        # moe_block.mlp.router keeps its own fresh random init from GPT.__init__ -- the
        # router has no dense analogue to copy from.
    return moe_model


# %% [markdown]
# ## 4. The training loop
#
# One function for both phases. For a `moe`-mode model it also runs the loss-free balancing
# update on a window (`balance_every` steps), not every step -- the session's "let the cook
# work for three hours, then go check" point -- and records each window's load-balance ratio
# (busiest expert's share of tokens / a perfectly even share) so the balancing mechanism's
# effect is visible in the results, not just asserted.

# %%
def reset_peak_memory():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mib():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


def train_run(model, cfg, batch_size, target_tokens, name, lr=3e-4, weight_decay=0.0,
              log_every=50, balance_every=8):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_steps = max(1, target_tokens // (batch_size * cfg.block_size))
    moe_blocks = [b.mlp for b in model.blocks if cfg.mode == "moe"]

    reset_peak_memory()
    losses, balance_ratios = [], []
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

        if moe_blocks and (step + 1) % balance_every == 0:
            ratios = []
            for mlp in moe_blocks:
                counts = mlp.maybe_rebalance()
                if counts is not None and counts.sum() > 0:
                    even_share = counts.sum() / mlp.n_experts
                    ratios.append(float(counts.max() / even_share))
            if ratios:
                balance_ratios.append({"step": step, "max_over_even": sum(ratios) / len(ratios)})

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
    thin = max(1, len(losses) // 400)
    result = {
        "mode": cfg.mode, "batch_size": batch_size, "steps": n_steps,
        "tokens_trained": tokens_seen, "final_loss": losses[-1], "min_loss": min(losses),
        "loss_curve": losses[::thin], "loss_curve_stride": thin,
        "wall_clock_s": wall, "tokens_per_s": tokens_seen / max(wall, 1e-9),
        "peak_memory_mib": peak_memory_mib(), "balance_ratios": balance_ratios,
    }
    RESULTS["runs"][name] = result
    return result, optimizer


# %% [markdown]
# ## 5. Run 1 — train the linear (dense) model
#
# A small decoder-only transformer with one `DenseMLP` per layer -- the "linear model" the
# assignment names, and section 14's own starting point before anything gets split into
# experts.

# %%
N_EMBD, N_HEAD, N_LAYER, MLP_MULT, BLOCK_SIZE = 192, 6, 4, 4, 128
BATCH_SIZE = 64
DENSE_TOKENS = 3_000_000
MOE_TOKENS = 6_000_000
N_EXPERTS, TOP_K = 8, 2
UPCYCLE_NOISE_STD = 0.02
BALANCE_GAMMA = 0.002
BALANCE_EVERY = 8

dense_cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=N_LAYER,
                       n_embd=N_EMBD, n_head=N_HEAD, mlp_mult=MLP_MULT, dropout=0.0,
                       mode="dense")
dense_model = GPT(dense_cfg).to(DEVICE)
dense_total, dense_active = dense_model.param_counts()
print(f"dense model: {dense_total:,} parameters")

dense_result, _ = train_run(dense_model, dense_cfg, BATCH_SIZE, DENSE_TOKENS, "dense")

# %% [markdown]
# ## 6. Convert to MoE, then keep training
#
# `n_experts=8`, `top_k=2`: every token activates 2 of 8 clones of what was one dense
# feed-forward network per layer. Training continues with a *fresh* optimizer (AdamW has no
# state for weights that didn't exist a moment ago) but the *same* model weights everywhere
# except the freshly initialized router and the noise just added to each expert clone.

# %%
moe_cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=N_LAYER,
                     n_embd=N_EMBD, n_head=N_HEAD, mlp_mult=MLP_MULT, dropout=0.0, mode="moe",
                     n_experts=N_EXPERTS, top_k=TOP_K, balance_gamma=BALANCE_GAMMA)
moe_model = upcycle_to_moe(dense_model, moe_cfg, noise_std=UPCYCLE_NOISE_STD, seed=SEED)
moe_total, moe_active = moe_model.param_counts()
print(f"moe model:   {moe_total:,} parameters total, {moe_active:,} active per token "
      f"({moe_active / moe_total:.1%})")
print(f"             {moe_total / dense_total:.2f}x the dense model's total parameters, "
      f"{moe_active / dense_total:.2f}x its active parameters")

# sanity check right at the seam: the loss just after conversion should be close to the dense
# model's final loss (same weights plus small noise + a fresh, nearly-uniform-at-init
# router), not a random-init-sized jump. Recorded, not asserted -- the real evidence is the
# training curve below.
moe_model.eval()
with torch.no_grad():
    x, y = get_batch(train_tokens, BATCH_SIZE, BLOCK_SIZE, DEVICE)
    _, seam_loss = moe_model(x, y)
moe_model.train()
RESULTS["seam_loss_right_after_conversion"] = float(seam_loss.item())
print(f"loss immediately after conversion (before any MoE-phase training): "
      f"{seam_loss.item():.4f}  (dense model's final training loss was "
      f"{dense_result['final_loss']:.4f})")

moe_result, _ = train_run(moe_model, moe_cfg, BATCH_SIZE, MOE_TOKENS, "moe",
                           balance_every=BALANCE_EVERY)

# %% [markdown]
# ## 7. Results
#
# Both runs, plus the parameter accounting and the balance-ratio trace, are written to
# `results/results.json`; `README.md`'s numbers are rendered from that file by
# `tools/render_numbers.py` and are never typed in by hand.

# %%
RESULTS["param_counts"] = {
    "dense_total": dense_total, "moe_total": moe_total, "moe_active": moe_active,
}
RESULTS["model_config"] = {
    "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER, "mlp_mult": MLP_MULT,
    "block_size": BLOCK_SIZE, "vocab_size": VOCAB_SIZE, "n_experts": N_EXPERTS, "top_k": TOP_K,
    "upcycle_noise_std": UPCYCLE_NOISE_STD, "balance_gamma": BALANCE_GAMMA,
    "balance_every": BALANCE_EVERY,
}

with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(RESULTS, fh, indent=2)
print(f"wrote {os.path.join(OUT_DIR, 'results.json')}")

# combined loss curve across both phases, with a marker at the conversion point
dense_xs = np.linspace(0, dense_result["tokens_trained"], len(dense_result["loss_curve"]))
moe_xs = dense_result["tokens_trained"] + np.linspace(
    0, moe_result["tokens_trained"], len(moe_result["loss_curve"]))

plt.figure(figsize=(8, 5))
plt.plot(dense_xs, dense_result["loss_curve"], label="dense (linear model)", color="tab:blue")
plt.plot(moe_xs, moe_result["loss_curve"], label="moe (8 experts, top-2)", color="tab:orange")
plt.axvline(dense_result["tokens_trained"], color="gray", linestyle="--", linewidth=1,
            label="upcycled to MoE here")
plt.xlabel("tokens trained")
plt.ylabel("training loss")
plt.legend()
plt.title("Session 14 - dense model grown into a mixture of experts")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=140)
plt.close()
print("wrote results/loss_curves.png")

if moe_result["balance_ratios"]:
    br_steps = [r["step"] for r in moe_result["balance_ratios"]]
    br_vals = [r["max_over_even"] for r in moe_result["balance_ratios"]]
    plt.figure(figsize=(8, 4))
    plt.plot(br_steps, br_vals, color="tab:green")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="perfectly even load")
    plt.xlabel("MoE-phase training step")
    plt.ylabel("busiest expert's load / even share")
    plt.legend()
    plt.title("Loss-free balancing: load-imbalance ratio over training")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "expert_balance.png"), dpi=140)
    plt.close()
    print("wrote results/expert_balance.png")

print("\n| run | mode | params (total/active) | steps | final loss | min loss | tok/s |")
print("|---|---|---|---|---|---|---|")
print(f"| dense | dense | {dense_total:,} / {dense_total:,} | {dense_result['steps']:,} | "
      f"{dense_result['final_loss']:.4f} | {dense_result['min_loss']:.4f} | "
      f"{dense_result['tokens_per_s']:,.0f} |")
print(f"| moe | moe | {moe_total:,} / {moe_active:,} | {moe_result['steps']:,} | "
      f"{moe_result['final_loss']:.4f} | {moe_result['min_loss']:.4f} | "
      f"{moe_result['tokens_per_s']:,.0f} |")
