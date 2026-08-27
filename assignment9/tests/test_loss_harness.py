"""Tests for the parts of the harness that a wrong answer would not announce.

`loss_harness.py` is a top-to-bottom script: importing it would train four models.  So the
pieces under test are lifted out of it with `ast` instead, which keeps a single source of
truth -- if the definitions in the notebook change, these tests change with them.
"""
import ast
import io
import os

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "loss_harness.py")

WANTED = ("ChunkedCrossEntropy", "chunked_cross_entropy", "ce_loss", "per_token_ce")


def load_pieces():
    """Compile only the named top-level definitions out of the harness."""
    tree = ast.parse(io.open(SRC, encoding="utf-8").read())
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in WANTED]
    missing = set(WANTED) - {n.name for n in keep}
    assert not missing, f"harness no longer defines {missing}"
    ns = {"torch": torch, "F": F, "IGNORE": -100}
    exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)
    return ns


PIECES = load_pieces()
chunked_cross_entropy = PIECES["chunked_cross_entropy"]
ce_loss = PIECES["ce_loss"]
per_token_ce = PIECES["per_token_ce"]

IGNORE = -100
V, D = 1024, 32


def fixture(n=512, masked_every=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, D, generator=g)
    w = torch.randn(V, D, generator=g) * 0.05
    t = torch.randint(0, V, (n,), generator=g)
    if masked_every:
        t[::masked_every] = IGNORE
    return h, w, t


@pytest.mark.parametrize("chunk", [1, 7, 64, 512, 4096])
@pytest.mark.parametrize("masked_every", [None, 3, 17])
def test_chunked_matches_ordinary(chunk, masked_every):
    """Same loss and same gradients as F.cross_entropy, for every chunk size."""
    h, w, t = fixture(masked_every=masked_every)

    ha, wa = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    ref = F.cross_entropy(ha @ wa.t(), t, ignore_index=IGNORE)
    ref.backward()

    hb, wb = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    got = chunked_cross_entropy(hb, wb, t, chunk_size=chunk, ignore_index=IGNORE)
    got.backward()

    assert torch.allclose(ref, got, atol=1e-6)
    assert torch.allclose(ha.grad, hb.grad, atol=1e-7)
    assert torch.allclose(wa.grad, wb.grad, atol=1e-7)


def test_chunk_boundary_can_be_entirely_masked():
    """A chunk in which every target is ignored must contribute nothing, not a NaN."""
    h, w, t = fixture(n=64)
    t[:16] = IGNORE                       # chunk 0 of 4 is completely dead
    hb, wb = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    loss = chunked_cross_entropy(hb, wb, t, chunk_size=16, ignore_index=IGNORE)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(hb.grad).all() and torch.isfinite(wb.grad).all()
    assert hb.grad[:16].abs().max() == 0, "masked positions must receive no gradient"


def test_gradient_is_softmax_minus_onehot():
    """Session 9 section 5: dL/dz = softmax(z) - onehot(y), and it sums to zero."""
    h, w, t = fixture(n=8)
    z = (h @ w.t()).requires_grad_(True)
    F.cross_entropy(z, t, reduction="sum").backward()
    expected = torch.softmax(z.detach(), dim=-1)
    expected[torch.arange(8), t] -= 1.0
    assert torch.allclose(z.grad, expected, atol=1e-6)
    assert z.grad.sum(dim=-1).abs().max() < 1e-5


def test_denominator_is_contributing_tokens_not_b_times_t():
    """ce_loss must divide by the number of non-ignored targets."""
    torch.manual_seed(1)
    logits = torch.randn(4, 10, V)
    targets = torch.randint(0, V, (4, 10))
    targets[:, 5:] = IGNORE
    loss, n = ce_loss(logits, targets, ignore_index=IGNORE)
    assert n == 20 and targets.numel() == 40
    terms = per_token_ce(logits, targets, ignore_index=IGNORE)
    assert torch.allclose(loss, terms.sum() / n, atol=1e-6)
    assert not torch.allclose(loss, terms.sum() / targets.numel(), atol=1e-4)


def test_masking_one_site_is_exactly_dropping_one_term():
    """The Item 4 arithmetic: masking removes one term from numerator and denominator."""
    torch.manual_seed(2)
    logits = torch.randn(1, 30, V)
    targets = torch.randint(0, V, (1, 30))
    before, n_before = ce_loss(logits, targets, ignore_index=IGNORE)
    terms = per_token_ce(logits, targets, ignore_index=IGNORE)
    masked = targets.clone()
    masked[0, 11] = IGNORE
    after, n_after = ce_loss(logits, masked, ignore_index=IGNORE)
    assert n_after == n_before - 1
    expected = (before.item() * n_before - terms[11].item()) / n_after
    assert abs(expected - after.item()) < 1e-4


def test_the_shift_is_in_the_right_direction():
    """inputs = tokens[:, :-1], targets = tokens[:, 1:] -- and target[i] follows input[i]."""
    tokens = torch.arange(20).view(2, 10)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert (targets - inputs == 1).all()
    assert not torch.equal(inputs, targets), "an unshifted target is a copy task"
