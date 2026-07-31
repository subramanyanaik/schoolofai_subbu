"""
P4 -- Mixture-transition gradient stability, executed.

README §10/§4.2 claims: an abrupt lane-share change spikes the gradient norm hard
enough to destabilise training, and a ramped transition prevents it. This script
tests that mechanism directly rather than asserting it.

Real inventory tokens are not required for this proxy -- the claim under test is
about *how a mixture-share transition is scheduled*, not about which languages
fill it, so two synthetic, statistically distinct token distributions stand in
for "lane A" and "lane B" (e.g. English-heavy S1 mix vs code-heavy S2 mix). This
is the same abstraction the full P4 design in proxy.py uses at 3B/60B scale --
here it runs on an actual GPU, at a size and step count a single consumer card
can finish, so the ramp-vs-step *mechanism* gets a real, executed answer instead
of a specified one.

Three conditions, matching proxy.py's P4 arms exactly:
  A  -- long ramp   (proportionally analogous to the 100B-token ramp)
  B  -- abrupt step (no ramp at all)
  C  -- short ramp  (proportionally analogous to the 20B-token ramp)

Metric: max gradient-norm multiplier over the transition window, relative to the
pre-transition steady-state median. Decision rule (unchanged from proxy.py):
keep the long ramp only if B spikes >=5x AND C also spikes; if C is sufficient,
the band was over-specified.
"""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- synthetic "lane" distributions -----------------------------------------
# Two disjoint-biased Markov chains over a shared small vocabulary stand in for
# two curriculum-stage mixes with genuinely different token statistics -- the
# property that actually drives a router/embedding gradient shock, not the
# specific languages involved.
VOCAB = 64
SEQ_LEN = 64


def make_markov(vocab, bias_strength, seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(vocab, vocab, generator=g)
    # sharpen each row so the chain has a strong, distinct, learnable "accent";
    # bias_strength > 1 pushes rows toward one-hot after normalization
    base = base.pow(bias_strength)
    return base / base.sum(dim=-1, keepdim=True)


LANE_A = make_markov(VOCAB, bias_strength=8.0, seed=1)   # e.g. "English-heavy S1"
LANE_B = make_markov(VOCAB, bias_strength=8.0, seed=2)   # e.g. "code-heavy S2"


def _sample_lane_batch(lane, n, seq_len, gen):
    """Vectorized: sample n sequences from one Markov lane at once."""
    if n == 0:
        return torch.empty(0, seq_len, dtype=torch.long)
    tok = torch.randint(0, VOCAB, (n,), generator=gen)
    out = [tok]
    for _ in range(seq_len - 1):
        probs = lane[tok]                      # (n, vocab)
        tok = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        out.append(tok)
    return torch.stack(out, dim=1)              # (n, seq_len)


def sample_batch(batch_size, seq_len, share_b, gen):
    """share_b in [0,1]: fraction of the batch drawn from LANE_B this step."""
    n_b = int(round(batch_size * share_b))
    n_a = batch_size - n_b
    xa = _sample_lane_batch(LANE_A, n_a, seq_len, gen)
    xb = _sample_lane_batch(LANE_B, n_b, seq_len, gen)
    x = torch.cat([xa, xb], dim=0)
    perm = torch.randperm(batch_size, generator=gen)
    return x[perm]


# ---- tiny transformer --------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=128, n_head=4, n_layer=4, seq_len=SEQ_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        b, t = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :t, :]
        h = self.blocks(h, mask=self.causal_mask[:t, :t])
        h = self.ln_f(h)
        return self.head(h)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---- one training run under a given transition schedule ---------------------
def run_condition(name, ramp_steps, total_steps=1000, transition_at=500,
                   batch_size=64, lr=3e-4, log_every=1):
    model = TinyGPT(VOCAB).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(hash(name) % (2**31))

    grad_norms = []
    losses = []
    for step in range(total_steps):
        # share_b: 0 before the transition, 1 after; ramps interpolate linearly
        # across [transition_at - ramp_steps/2, transition_at + ramp_steps/2],
        # matching the plan's "ramp straddles the boundary" rule (§4.2).
        if ramp_steps == 0:
            share_b = 0.0 if step < transition_at else 1.0
        else:
            half = ramp_steps / 2
            lo, hi = transition_at - half, transition_at + half
            if step < lo:
                share_b = 0.0
            elif step > hi:
                share_b = 1.0
            else:
                share_b = (step - lo) / (hi - lo)

        x = sample_batch(batch_size, SEQ_LEN, share_b, gen).to(DEVICE)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))

        opt.zero_grad()
        loss.backward()
        total_norm = torch.norm(
            torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
        ).item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)  # measure only, don't actually clip meaningfully
        opt.step()

        grad_norms.append(total_norm)
        losses.append(loss.item())

    return dict(name=name, grad_norms=grad_norms, losses=losses,
                total_steps=total_steps, transition_at=transition_at, ramp_steps=ramp_steps)


def summarize(run, window=60):
    gn = run["grad_norms"]
    t0 = run["transition_at"]
    half_window = window // 2
    baseline = gn[max(0, t0 - 200):t0 - half_window]
    baseline_med = sorted(baseline)[len(baseline) // 2]
    transition_window = gn[max(0, t0 - half_window):t0 + half_window + run["ramp_steps"] // 2 + 40]
    peak = max(transition_window)
    multiplier = peak / baseline_med if baseline_med > 0 else float("inf")

    losses = run["losses"]
    loss_baseline = sorted(losses[max(0, t0 - 200):t0 - half_window])
    loss_med = loss_baseline[len(loss_baseline) // 2]
    spike_count = sum(
        1 for l in losses[max(0, t0 - half_window):t0 + half_window + run["ramp_steps"] // 2 + 40]
        if l > 1.5 * loss_med
    )
    return dict(
        name=run["name"], ramp_steps=run["ramp_steps"],
        baseline_median_grad_norm=baseline_med, peak_grad_norm=peak,
        grad_norm_multiplier=multiplier, loss_spike_count=spike_count,
    )


if __name__ == "__main__":
    t_start = time.time()
    model_probe = TinyGPT(VOCAB)
    print(f"model params: {n_params(model_probe):,}  device: {DEVICE}")

    conditions = [
        ("B_abrupt_step", 0),
        ("C_short_ramp", 60),     # proportionally analogous to the 20B-token ramp
        ("A_long_ramp", 300),     # proportionally analogous to the 100B-token ramp
    ]

    results = []
    for name, ramp in conditions:
        print(f"running {name} (ramp_steps={ramp}) ...")
        run = run_condition(name, ramp_steps=ramp)
        summ = summarize(run)
        results.append(dict(run=run, summary=summ))
        print(f"  -> peak/baseline multiplier = {summ['grad_norm_multiplier']:.2f}x, "
              f"loss spikes = {summ['loss_spike_count']}")

    out = {
        "device": DEVICE,
        "model_params": n_params(model_probe),
        "vocab": VOCAB, "seq_len": SEQ_LEN,
        "wall_clock_seconds": time.time() - t_start,
        "summaries": [r["summary"] for r in results],
    }
    Path("results.json").write_text(json.dumps(out, indent=2))

    # raw per-step traces too, for anyone who wants to re-plot
    traces = {r["run"]["name"]: {"grad_norms": r["run"]["grad_norms"], "losses": r["run"]["losses"]}
              for r in results}
    Path("traces.json").write_text(json.dumps(traces))

    print(json.dumps(out, indent=2))
