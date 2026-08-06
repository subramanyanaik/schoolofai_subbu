"""
P4 v2 -- Mixture-transition gradient stability, deepened.

v1 (run_p4.py) established the mechanism on synthetic Markov-chain "lanes":
an abrupt lane-share transition shocks the gradient, and ramping suppresses
it, with the shock shrinking monotonically as ramp length grows.

v2 asks a sharper question with a stronger setup:
  1. Real text instead of synthetic distributions -- lane A is this repo's
     own prose (README.md, SPECIFICATION.md, SETUP.md), lane B is this
     repo's own Python source (config.py, budget.py, mixture.py, ...). This
     is a genuine, naturally-occurring analogue of the plan's actual
     S1 (English-heavy) -> S2 (code-heavy) transition, not an abstraction
     standing in for it.
  2. A larger model (~2.7M params vs v1's 818K).
  3. Multiple seeds per ramp length, so the reported numbers are means with
     spread, not single noisy runs.
  4. A fine-grained ramp-length sweep instead of 3 discrete points, to find
     the actual minimum ramp length that suppresses the spike below
     threshold -- a specific number, not "somewhere between 60 and 300."
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- real text corpora -------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # assignment5/

PROSE_FILES = ["README.md", "docs/SPECIFICATION.md"]
CODE_FILES = [f"src/erav5/{m}.py" for m in
              ("config", "budget", "mixture", "curriculum", "floors",
               "proxy", "sensitivity", "validate", "export", "run_plan")]

# The corpus is FROZEN into corpus_prose.txt / corpus_code.txt alongside this
# script, and those snapshots are what the run reads. This matters for more than
# tidiness: README.md is itself part of the prose corpus AND is where this
# experiment's results are written up, so rebuilding the corpus from the live
# repo on every run would make the input depend on the previous run's output.
# Freezing the snapshot breaks that loop and makes the numbers below
# reproducible from a clean clone indefinitely. Pass --refresh-corpus to
# regenerate the snapshots from the current repo (which changes the inputs, and
# therefore legitimately changes the numbers).
PROSE_SNAPSHOT = HERE / "corpus_prose.txt"
CODE_SNAPSHOT = HERE / "corpus_code.txt"


def _build_corpus():
    prose = "".join((REPO / f).read_text(encoding="utf-8") for f in PROSE_FILES)
    code = "".join((REPO / f).read_text(encoding="utf-8") for f in CODE_FILES)
    return prose, code


if "--refresh-corpus" in sys.argv or not (PROSE_SNAPSHOT.exists() and CODE_SNAPSHOT.exists()):
    PROSE_TEXT, CODE_TEXT = _build_corpus()
    PROSE_SNAPSHOT.write_text(PROSE_TEXT, encoding="utf-8")
    CODE_SNAPSHOT.write_text(CODE_TEXT, encoding="utf-8")
else:
    PROSE_TEXT = PROSE_SNAPSHOT.read_text(encoding="utf-8")
    CODE_TEXT = CODE_SNAPSHOT.read_text(encoding="utf-8")

VOCAB_CHARS = sorted(set(PROSE_TEXT + CODE_TEXT))
VOCAB = len(VOCAB_CHARS)
STOI = {c: i for i, c in enumerate(VOCAB_CHARS)}


def encode(text):
    return torch.tensor([STOI[c] for c in text], dtype=torch.long)


PROSE_IDS = encode(PROSE_TEXT)   # lane A -- real English prose from this repo
CODE_IDS = encode(CODE_TEXT)     # lane B -- real Python source from this repo

SEQ_LEN = 96


def _sample_lane_batch(ids, n, seq_len, gen):
    if n == 0:
        return torch.empty(0, seq_len, dtype=torch.long)
    max_start = len(ids) - seq_len - 1
    starts = torch.randint(0, max_start, (n,), generator=gen)
    return torch.stack([ids[s:s + seq_len] for s in starts.tolist()], dim=0)


def sample_batch(batch_size, seq_len, share_b, gen):
    """share_b in [0,1]: fraction of the batch drawn from the CODE corpus."""
    n_b = int(round(batch_size * share_b))
    n_a = batch_size - n_b
    xa = _sample_lane_batch(PROSE_IDS, n_a, seq_len, gen)
    xb = _sample_lane_batch(CODE_IDS, n_b, seq_len, gen)
    x = torch.cat([xa, xb], dim=0)
    perm = torch.randperm(batch_size, generator=gen)
    return x[perm]


# ---- larger transformer -------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=192, n_head=6, n_layer=6, seq_len=SEQ_LEN, ff_mult=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=ff_mult * d_model,
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
def run_condition(seed, ramp_steps, total_steps=800, transition_at=500,
                   batch_size=48, lr=3e-4):
    torch.manual_seed(seed)
    model = TinyGPT(VOCAB).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    grad_norms = []
    losses = []
    for step in range(total_steps):
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
        opt.step()

        grad_norms.append(total_norm)
        losses.append(loss.item())

    return dict(grad_norms=grad_norms, losses=losses,
                total_steps=total_steps, transition_at=transition_at, ramp_steps=ramp_steps)


def summarize(run, window=80):
    gn = run["grad_norms"]
    t0 = run["transition_at"]
    half_window = window // 2
    baseline = gn[max(0, t0 - 300):t0 - half_window]
    baseline_med = sorted(baseline)[len(baseline) // 2]
    tail = t0 + half_window + run["ramp_steps"] // 2 + 60
    transition_window = gn[max(0, t0 - half_window):tail]
    peak = max(transition_window)
    multiplier = peak / baseline_med if baseline_med > 0 else float("inf")

    losses = run["losses"]
    loss_baseline = sorted(losses[max(0, t0 - 300):t0 - half_window])
    loss_med = loss_baseline[len(loss_baseline) // 2]
    spike_count = sum(1 for l in losses[max(0, t0 - half_window):tail] if l > 1.5 * loss_med)
    return dict(grad_norm_multiplier=multiplier, loss_spike_count=spike_count,
                baseline_median_grad_norm=baseline_med, peak_grad_norm=peak)


if __name__ == "__main__":
    t_start = time.time()
    probe = TinyGPT(VOCAB)
    print(f"vocab: {VOCAB}  model params: {n_params(probe):,}  device: {DEVICE}")
    print(f"prose corpus: {len(PROSE_IDS):,} chars   code corpus: {len(CODE_IDS):,} chars")

    RAMP_SWEEP = [0, 20, 40, 60, 100, 150, 200, 300]
    SEEDS = [0, 1, 2]
    SPIKE_THRESHOLD = 1.5  # grad-norm multiplier considered a "spike"

    all_results = {}
    for ramp in RAMP_SWEEP:
        mults, spikes = [], []
        for seed in SEEDS:
            run = run_condition(seed=seed * 1000 + ramp, ramp_steps=ramp)
            s = summarize(run)
            mults.append(s["grad_norm_multiplier"])
            spikes.append(s["loss_spike_count"])
        mean_mult = statistics.mean(mults)
        std_mult = statistics.pstdev(mults) if len(mults) > 1 else 0.0
        mean_spikes = statistics.mean(spikes)
        all_results[ramp] = dict(
            ramp_steps=ramp, seeds=SEEDS,
            multipliers=mults, mean_multiplier=mean_mult, std_multiplier=std_mult,
            spike_counts=spikes, mean_spike_count=mean_spikes,
        )
        print(f"ramp={ramp:>4} steps -> mean multiplier {mean_mult:.3f}x "
              f"(+/-{std_mult:.3f}, n={len(SEEDS)}), mean loss spikes {mean_spikes:.1f}")

    # find the minimum ramp length whose mean multiplier drops below threshold
    threshold_ramp = None
    for ramp in RAMP_SWEEP:
        if all_results[ramp]["mean_multiplier"] < SPIKE_THRESHOLD:
            threshold_ramp = ramp
            break

    out = {
        "device": DEVICE,
        "model_params": n_params(probe),
        "vocab_size": VOCAB,
        "seq_len": SEQ_LEN,
        "prose_corpus_chars": len(PROSE_IDS),
        "code_corpus_chars": len(CODE_IDS),
        "ramp_sweep": RAMP_SWEEP,
        "seeds_per_point": SEEDS,
        "spike_threshold": SPIKE_THRESHOLD,
        "minimum_ramp_under_threshold": threshold_ramp,
        "results_by_ramp": all_results,
        "wall_clock_seconds": time.time() - t_start,
    }
    Path("results_v2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "results_by_ramp"}, indent=2))
    print(f"\nminimum ramp length with mean multiplier < {SPIKE_THRESHOLD}x: {threshold_ramp}")
