"""Tests for the parts of the harness where a wrong answer would look like a right one.

`truth_harness.py` is a top-to-bottom script: importing it would train seven models.  So the
pieces under test are lifted out of it with `ast`, which keeps one source of truth -- if a
definition in the notebook changes, these tests change with it, and a stale copy cannot pass.
"""
import ast
import io
import math
import os
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "truth_harness.py")

WANTED = ("rne", "encode", "props", "lagged_corr", "robust_z", "accumulate",
          "global_grad_norm", "digits_agreeing", "cast_error")


def load_pieces():
    tree = ast.parse(io.open(SRC, encoding="utf-8").read())
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in WANTED]
    missing = set(WANTED) - {n.name for n in keep}
    assert not missing, f"harness no longer defines {missing}"
    ns = {"torch": torch, "F": F, "np": np, "math": math, "Fraction": Fraction}
    exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)
    return ns


P = load_pieces()
rne, encode, props = P["rne"], P["encode"], P["props"]
lagged_corr, robust_z = P["lagged_corr"], P["robust_z"]
accumulate, global_grad_norm = P["accumulate"], P["global_grad_norm"]
digits_agreeing, cast_error = P["digits_agreeing"], P["cast_error"]

FP32 = (8, 23, 127, True)
BF16 = (8, 7, 127, True)
FP16 = (5, 10, 15, True)
E4M3 = (4, 3, 7, False)


# --------------------------------------------------------------------------- number formats

def test_rne_ties_go_to_even():
    assert rne(Fraction(1, 2)) == 0            # 0.5 -> 0, not 1
    assert rne(Fraction(3, 2)) == 2
    assert rne(Fraction(5, 2)) == 2
    assert rne(Fraction(7, 2)) == 4
    assert rne(Fraction(1, 3)) == 0
    assert rne(Fraction(2, 3)) == 1


def test_the_answer_for_0p1_is_the_one_written_in_the_readme():
    """The three bit patterns the assignment asks for, spelled out independently here."""
    assert encode(Fraction(1, 10), *FP32, "fp32")["bits"] == \
        "00111101110011001100110011001101"
    assert encode(Fraction(1, 10), *BF16, "bf16")["bits"] == "0011110111001101"
    assert encode(Fraction(1, 10), *E4M3, "fp8")["bits"] == "00011101"
    # ...and none of them is 0.1.
    for fmt in (FP32, BF16, E4M3):
        assert encode(Fraction(1, 10), *fmt, "x")["exact"] != Fraction(1, 10)


def test_0p1_decoded_values():
    """What each format actually stores when you ask it for 0.1.  None of them is 0.1."""
    assert encode(Fraction(1, 10), *FP32, "fp32")["exact"] == Fraction(13421773, 2 ** 27)
    assert float(encode(Fraction(1, 10), *FP32, "fp32")["exact"]) == 0.100000001490116119
    assert float(encode(Fraction(1, 10), *BF16, "bf16")["exact"]) == 0.10009765625
    assert float(encode(Fraction(1, 10), *E4M3, "fp8")["exact"]) == 0.1015625
    # and the double the interpreter itself hands you is not 0.1 either
    assert Fraction(0.1) != Fraction(1, 10)


@pytest.mark.parametrize("v", [0.1, 1.0, -1.0, 3.14159, 1e-3, -2.5e-4, 448.0, 6e-8,
                               1.0 / 3, 65504.0, 2.0 ** -9, 2.0 ** -6, 1e-30])
def test_encode_matches_torch_bf16(v):
    t = torch.tensor([v], dtype=torch.float32)
    mine = encode(Fraction(float(t.item())), *BF16, "bf16")["bits"]
    theirs = format(int(t.to(torch.bfloat16).view(torch.int16).item()) & 0xFFFF, "016b")
    assert mine == theirs


@pytest.mark.parametrize("v", [0.1, 1.0, -1.0, 0.375, 1e-3, 2.0 ** -9, 2.0 ** -6,
                               2.0 ** -8, 448.0, -448.0, 0.0078125, 5.5])
def test_encode_matches_torch_fp8_e4m3(v):
    t = torch.tensor([v], dtype=torch.float32)
    mine = encode(Fraction(float(t.item())), *E4M3, "fp8")["bits"]
    theirs = format(int(t.to(torch.float8_e4m3fn).view(torch.uint8).item()), "08b")
    assert mine == theirs


def test_encode_matches_torch_over_a_random_sweep():
    g = torch.Generator().manual_seed(11)
    vals = torch.cat([torch.randn(1500, generator=g),
                      torch.randn(500, generator=g) * 1e-4]).float()
    for v in vals.tolist():
        if v == 0:
            continue
        t = torch.tensor([v], dtype=torch.float32)
        exact = Fraction(float(t.item()))
        assert encode(exact, *BF16, "bf16")["bits"] == \
            format(int(t.to(torch.bfloat16).view(torch.int16).item()) & 0xFFFF, "016b")
        assert encode(exact, *FP16, "fp16")["bits"] == \
            format(int(t.to(torch.float16).view(torch.int16).item()) & 0xFFFF, "016b")


def test_props_known_constants():
    fp32 = props(*FP32[:3])
    assert fp32["min_normal"] == pytest.approx(1.1754943508222875e-38)
    assert fp32["max_val"] == pytest.approx(3.4028234663852886e38)
    bf16 = props(*BF16[:3])
    assert bf16["min_normal"] == fp32["min_normal"], "bf16 keeps fp32's range"
    assert bf16["eps"] == 2.0 ** -7
    fp16 = props(*FP16[:3])
    assert fp16["min_normal"] == pytest.approx(6.103515625e-05)
    assert fp16["max_val"] == pytest.approx(65504.0)
    e4m3 = props(4, 3, 7, False)
    assert e4m3["max_val"] == pytest.approx(448.0), "E4M3FN spends the top exponent"
    assert e4m3["min_normal"] == pytest.approx(0.015625)
    assert e4m3["min_subnormal"] == pytest.approx(0.001953125)


# ------------------------------------------------------------------- gradient accumulation

class ToyLM(nn.Module):
    """Smallest thing with the harness's forward signature: (logits, loss)."""

    def __init__(self, vocab=7, dim=5):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx, targets=None, tr=None, reduction="mean"):
        logits = self.head(self.emb(idx))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100,
                                   reduction=reduction)
        return logits, loss


def ragged_micros(lengths, vocab=7, T=6, B=2, seed=0):
    """Micro-batches whose contributing-token counts differ, plus the same data flattened."""
    g = torch.Generator().manual_seed(seed)
    micros, xs, ys = [], [], []
    for L in lengths:
        x = torch.randint(0, vocab, (B, T), generator=g)
        y = torch.randint(0, vocab, (B, T), generator=g)
        y[:, L:] = -100
        micros.append((x, y, int((y != -100).sum())))
        xs.append(x)
        ys.append(y)
    return micros, torch.cat(xs), torch.cat(ys)


def grad_vector(m):
    return torch.cat([p.grad.reshape(-1) for p in m.parameters()])


def test_correct_accumulation_reproduces_the_full_batch_gradient():
    """The definition of correct: accumulating must equal not having accumulated at all."""
    torch.manual_seed(0)
    m = ToyLM()
    micros, X, Y = ragged_micros([6, 4, 1])

    accumulate(m, micros, "correct")
    g_acc = grad_vector(m).clone()

    m.zero_grad(set_to_none=True)
    _, loss = m(X, Y, reduction="mean")            # one batch, no accumulation
    loss.backward()
    g_full = grad_vector(m)

    assert torch.allclose(g_acc, g_full, atol=1e-6), \
        "token-weighted accumulation must be exactly the un-accumulated gradient"


def test_broken_accumulation_does_not():
    torch.manual_seed(0)
    m = ToyLM()
    micros, X, Y = ragged_micros([6, 4, 1])

    accumulate(m, micros, "broken")
    g_bad = grad_vector(m).clone()

    m.zero_grad(set_to_none=True)
    _, loss = m(X, Y, reduction="mean")
    loss.backward()
    g_full = grad_vector(m)

    assert not torch.allclose(g_bad, g_full, atol=1e-4)
    cos = F.cosine_similarity(g_bad, g_full, dim=0).item()
    assert cos < 0.9999, f"the broken rule should not be a rounding error away ({cos})"


def test_the_two_rules_coincide_when_lengths_are_equal():
    """Why it went unnoticed until 2024: with fixed-length packing there is no bug."""
    torch.manual_seed(0)
    m = ToyLM()
    micros, _, _ = ragged_micros([6, 6, 6])

    accumulate(m, micros, "correct")
    a = grad_vector(m).clone()
    accumulate(m, micros, "broken")
    b = grad_vector(m).clone()
    assert torch.allclose(a, b, atol=1e-9)


def test_accumulation_weights_are_the_token_shares():
    """Spot-check the arithmetic itself: correct == sum(n_i/N * mean_i)."""
    torch.manual_seed(1)
    m = ToyLM()
    micros, _, _ = ragged_micros([6, 3, 2], seed=3)
    N = sum(n for _, _, n in micros)

    accumulate(m, micros, "correct")
    got = grad_vector(m).clone()

    want = torch.zeros_like(got)
    for x, y, n in micros:
        m.zero_grad(set_to_none=True)
        _, l = m(x, y, reduction="mean")
        l.backward()
        want += grad_vector(m) * (n / N)
    assert torch.allclose(got, want, atol=1e-6)


# ------------------------------------------------------------------------- the step logger

def test_global_grad_norm_is_what_clip_grad_norm_reports():
    torch.manual_seed(0)
    m = ToyLM()
    x = torch.randint(0, 7, (2, 6))
    _, l = m(x, x)
    l.backward()
    mine = global_grad_norm(m)
    theirs = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9).item()
    assert mine == pytest.approx(theirs, rel=1e-6)


def test_lagged_corr_recovers_a_known_lead():
    """b is a delayed copy of a: the peak correlation must sit at exactly that delay."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    lead = 3
    b = np.concatenate([np.zeros(lead), a[:-lead]]) + 0.01 * rng.normal(size=400)
    lags = list(range(-8, 9))
    cc = lagged_corr(a, b, lags)
    assert lags[int(np.argmax(cc))] == lead


def test_robust_z_is_robust():
    series = [1.0] * 30 + [50.0]
    z, med, mad = robust_z(series, 30, window=25)
    assert med == 1.0
    assert z > 10
    z2, _, _ = robust_z(series, 20, window=25)
    assert abs(z2) < 1e-9, "a quiet step must not be flagged"


# --------------------------------------------------------------------------------- helpers

def test_digits_agreeing():
    assert digits_agreeing(1.0, 1.0) == float("inf")
    assert digits_agreeing(1.001, 1.0) == pytest.approx(3.0, abs=0.01)
    assert digits_agreeing(1.1, 1.0) == pytest.approx(1.0, abs=0.01)


def test_cast_error_ranks_the_formats_the_way_the_writeup_claims():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(20000, generator=g) * 1e-3
    bf16 = cast_error(t, torch.bfloat16)
    fp8 = cast_error(t, torch.float8_e4m3fn)
    fp8s = cast_error(t, torch.float8_e4m3fn, scale=448.0 / t.abs().max().item())
    assert bf16["rel"] < fp8s["rel"] < fp8["rel"]
    assert fp8["zeroed"] > 0, "unscaled fp8 must flush small values to zero"
    assert bf16["zeroed"] == 0
