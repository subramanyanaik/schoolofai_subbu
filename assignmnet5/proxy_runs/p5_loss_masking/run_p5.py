"""
P5 -- Loss-masking on tool observations, executed.

README §2.1's headline claim: "tool observations are context-only -- zero
loss." Only model-generated tokens receive gradient; training on tool
responses teaches the model to hallucinate observations. This script tests
the mechanism directly with a synthetic tool-use task small enough for a
single consumer GPU, mirroring proxy.py's P5 arms exactly:

  Arm A -- loss on model-generated tokens only (tool output span masked)
  Arm B -- loss on the full trajectory including the tool-output span

Synthetic task: single-digit addition via a fixed-format trajectory
  Q{a}{b} C{a}{b} O{sum:02d} A{parity}
  ^prompt  ^tool   ^tool      ^final
           call    output     answer (model-generated)

"O{sum}" is the tool's response -- deterministic, environment-provided,
exactly analogous to a real tool's return value. "A{parity}" is the
downstream answer the model must produce FROM that tool output (even/odd of
the sum), standing in for actually using a tool result rather than just
restating it.

Two things get measured, matching the plan's own decision rule
("A must cut fabricated-observation rate by >=50% at no task-success cost"):

  1. Fabrication rate: with the true tool output withheld, how often does
     the model spontaneously generate the *correct* tool response anyway --
     i.e. does it behave as if it can compute the tool's job itself?
  2. Task success: with the true tool output supplied, does the model
     correctly read it and produce the right final answer? This checks that
     masking the loss doesn't cost anything on the task the tool exists for.
"""
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- synthetic tool-trajectory task ------------------------------------------
VOCAB_CHARS = sorted(set("QCOADE0123456789"))
VOCAB = len(VOCAB_CHARS)
STOI = {c: i for i, c in enumerate(VOCAB_CHARS)}


def seq_for(a, b):
    s = a + b
    parity = "E" if s % 2 == 0 else "D"
    return f"Q{a}{b}C{a}{b}O{s:02d}A{parity}"


SEQ_LEN = len(seq_for(0, 0))  # 11: Q a b C a b O d d A p

# positions (0-indexed) of the tool-output span "O" + 2 digits
OUTPUT_SPAN = {6, 7, 8}

ALL_COMBOS = [(a, b) for a in range(10) for b in range(10)]  # 100 combos
random.Random(0).shuffle(ALL_COMBOS)
HOLD_OUT = ALL_COMBOS[:20]
TRAIN = ALL_COMBOS[20:]


def encode(s):
    return torch.tensor([STOI[c] for c in s], dtype=torch.long)


TRAIN_SEQS = torch.stack([encode(seq_for(a, b)) for a, b in TRAIN])  # (80, 11)


def sample_batch(batch_size, gen):
    idx = torch.randint(0, len(TRAIN_SEQS), (batch_size,), generator=gen)
    return TRAIN_SEQS[idx]


# ---- tiny transformer ---------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=64, n_head=4, n_layer=3, seq_len=SEQ_LEN, ff_mult=4):
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


def masked_loss(logits, targets, mask_output):
    """targets[i] is the original-sequence character at position i+1
    (predicted from logits[i]). mask_output=True zeroes loss for target
    positions inside OUTPUT_SPAN -- the tool's response is context, not a
    generation target."""
    B, T, V = logits.shape
    per_tok = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none").reshape(B, T)
    if not mask_output:
        return per_tok.mean()
    weight = torch.ones(T, device=logits.device)
    for orig_pos in OUTPUT_SPAN:
        weight[orig_pos - 1] = 0.0   # target index t predicts original position t+1
    return (per_tok * weight.unsqueeze(0)).sum() / (weight.sum() * B)


def run_arm(seed, mask_output, steps, batch_size=64, lr=3e-4):
    torch.manual_seed(seed)
    model = TinyGPT(VOCAB).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        x = sample_batch(batch_size, gen).to(DEVICE)
        logits = model(x[:, :-1])
        loss = masked_loss(logits, x[:, 1:], mask_output)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


# ---- evals ---------------------------------------------------------------------
@torch.no_grad()
def eval_fabrication(model, combos):
    """Withhold the true tool output; let the model generate freely past the
    tool-call span and check how often it produces the CORRECT output anyway."""
    model.eval()
    correct = 0
    for a, b in combos:
        cur = encode(f"Q{a}{b}C{a}{b}").unsqueeze(0).to(DEVICE)
        gen_chars = []
        for _ in range(3):  # O + 2 digits
            logits = model(cur)
            nxt = logits[0, -1].argmax().item()
            gen_chars.append(VOCAB_CHARS[nxt])
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        true_o = f"O{(a + b):02d}"
        if "".join(gen_chars) == true_o:
            correct += 1
    model.train()
    return correct / len(combos)


@torch.no_grad()
def eval_task_success(model, combos):
    """Supply the TRUE tool output; check whether the model reads it correctly
    and produces the right final answer."""
    model.eval()
    correct = 0
    for a, b in combos:
        s = a + b
        cur = encode(f"Q{a}{b}C{a}{b}O{s:02d}").unsqueeze(0).to(DEVICE)
        gen_chars = []
        for _ in range(2):  # A + parity
            logits = model(cur)
            nxt = logits[0, -1].argmax().item()
            gen_chars.append(VOCAB_CHARS[nxt])
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        parity = "E" if s % 2 == 0 else "D"
        if "".join(gen_chars) == f"A{parity}":
            correct += 1
    model.train()
    return correct / len(combos)


def run_condition(name, mask_output, steps, seeds):
    fab_train, fab_hold, task_train, task_hold = [], [], [], []
    for seed in seeds:
        model, losses = run_arm(seed=seed, mask_output=mask_output, steps=steps)
        fab_train.append(eval_fabrication(model, TRAIN))
        fab_hold.append(eval_fabrication(model, HOLD_OUT))
        task_train.append(eval_task_success(model, TRAIN))
        task_hold.append(eval_task_success(model, HOLD_OUT))
    def m(xs):
        return sum(xs) / len(xs)
    return dict(
        name=name, mask_output=mask_output, steps=steps, seeds=seeds,
        fabrication_rate_train=m(fab_train), fabrication_rate_holdout=m(fab_hold),
        task_success_train=m(task_train), task_success_holdout=m(task_hold),
        fabrication_rate_train_runs=fab_train, fabrication_rate_holdout_runs=fab_hold,
        task_success_train_runs=task_train, task_success_holdout_runs=task_hold,
        final_train_loss=losses[-1],
    )


if __name__ == "__main__":
    t0 = time.time()
    probe = TinyGPT(VOCAB)
    print(f"vocab: {VOCAB}  seq_len: {SEQ_LEN}  model params: {n_params(probe):,}  device: {DEVICE}")
    print(f"train combos: {len(TRAIN)}  held-out combos: {len(HOLD_OUT)}")

    STEPS = 4000
    SEEDS = [0, 1, 2]

    results = {}
    for name, mask in [("A_masked_tool_output", True), ("B_full_loss", False)]:
        print(f"running {name} (mask_output={mask}) ...")
        r = run_condition(name, mask, STEPS, SEEDS)
        results[name] = r
        print(f"  fabrication: train={r['fabrication_rate_train']:.2f} holdout={r['fabrication_rate_holdout']:.2f}"
              f"   task-success: train={r['task_success_train']:.2f} holdout={r['task_success_holdout']:.2f}"
              f"   final_loss={r['final_train_loss']:.4f}")

    out = {
        "device": DEVICE, "model_params": n_params(probe), "vocab_size": VOCAB,
        "seq_len": SEQ_LEN, "steps_per_arm": STEPS, "seeds": SEEDS,
        "n_train_combos": len(TRAIN), "n_holdout_combos": len(HOLD_OUT),
        "results": results,
        "wall_clock_seconds": time.time() - t0,
    }
    Path("results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
