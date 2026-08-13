"""E2 -- the acceptance gate that carries the evidentiary weight.

KAS is an *exact reformulation* of the dense full-softmax head, not an
approximation of it.  Everything here is arithmetic, so it must hold on any
machine: logits, the log-partition function and gradients all match the naive
dense head to <1e-10 in fp64 and <1e-4 in fp32, for every implementation, tied
and untied, with and without the unigram bias.
"""

import numpy as np
import pytest
import torch

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.kas_head import KASHead, Occupancy, dense_codec_table
from kronecker_v2.vocab import PackedVocab

FP64_TOL = 1e-10
FP32_TOL = 1e-4


def toy_vocab(V=512, L_max=32, seed=0, max_len=None):
    rng = np.random.default_rng(seed)
    hi = max_len or L_max
    toks = [
        bytes(rng.integers(0, 256, size=int(n), dtype=np.uint8).tolist())
        for n in rng.integers(1, hi + 1, size=V)
    ]
    return PackedVocab(toks, L_max=L_max)


def make(d_c=32, L_max=32, V=512, d_model=48, dtype=torch.float64, seed=0, **kw):
    codec = KroneckerCodec(d_c=d_c, L_max=L_max, seed=seed)
    packed = toy_vocab(V=V, L_max=L_max, seed=seed)
    torch.manual_seed(seed)
    head = KASHead(d_model, codec, packed, dtype=dtype, **kw)
    table = dense_codec_table(packed, codec, dtype=dtype)
    head.attach_dense_table(table)
    return codec, packed, head, table


# --------------------------------------------------------------------- logits


@pytest.mark.parametrize("d_c", [16, 32, 64])
def test_logits_match_dense_fp64(d_c):
    _, _, head, table = make(d_c=d_c)
    h = torch.randn(7, 48, dtype=torch.float64)
    ref = head.dense_logits(h, table)
    for impl in ("bag", "sparse", "loop", "dense"):
        got = head(h, impl=impl)
        err = (got - ref).abs().max().item()
        assert err < FP64_TOL, f"{impl}: {err}"


def test_all_implementations_agree():
    _, _, head, _ = make()
    h = torch.randn(5, 48, dtype=torch.float64)
    outs = {impl: head(h, impl=impl) for impl in ("bag", "sparse", "loop", "dense")}
    base = outs["bag"]
    for impl, got in outs.items():
        assert (got - base).abs().max().item() < FP64_TOL, impl


def test_logits_match_dense_fp32():
    _, _, head, table = make(dtype=torch.float32)
    h = torch.randn(7, 48, dtype=torch.float32)
    ref = head.dense_logits(h, table)
    for impl in ("bag", "sparse", "loop"):
        err = (head(h, impl=impl) - ref).abs().max().item()
        assert err < FP32_TOL, f"{impl}: {err}"


# ------------------------------------------------------------ partition + grad


def test_log_partition_function_matches():
    _, _, head, table = make()
    h = torch.randn(9, 48, dtype=torch.float64)
    ref = torch.logsumexp(head.dense_logits(h, table), dim=-1)
    got = torch.logsumexp(head(h, impl="bag"), dim=-1)
    assert (got - ref).abs().max().item() < FP64_TOL


def test_gradients_match_dense():
    codec, packed, head, table = make()
    h = torch.randn(11, 48, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, packed.V, (11,))

    def loss_of(logits):
        return torch.nn.functional.cross_entropy(logits, targets)

    grads = {}
    for name, logits in (
        ("dense", head.dense_logits(h, table)),
        ("kas", head(h, impl="bag")),
    ):
        head.zero_grad(set_to_none=True)
        if h.grad is not None:
            h.grad = None
        loss_of(logits).backward()
        grads[name] = (
            head.W_out.grad.clone(),
            head.beta.grad.clone(),
            h.grad.clone(),
        )

    for a, b in zip(grads["dense"], grads["kas"]):
        assert (a - b).abs().max().item() < FP64_TOL


def test_cross_entropy_loss_matches():
    _, packed, head, table = make()
    h = torch.randn(13, 48, dtype=torch.float64)
    t = torch.randint(0, packed.V, (13,))
    ce = torch.nn.functional.cross_entropy
    ref = ce(head.dense_logits(h, table), t)
    got = ce(head(h, impl="bag"), t)
    assert (ref - got).abs().item() < FP64_TOL


# ----------------------------------------------------------------- variants


def test_tied_path_is_exact():
    codec = KroneckerCodec(d_c=32, L_max=32, seed=0)
    packed = toy_vocab(L_max=32)
    tied = torch.nn.Parameter(torch.randn(codec.D, 48, dtype=torch.float64) * 0.02)
    head = KASHead(48, codec, packed, tied_weight=tied, dtype=torch.float64)
    table = dense_codec_table(packed, codec)
    h = torch.randn(6, 48, dtype=torch.float64)
    assert head.is_tied and head.W_out is None
    err = (head(h, impl="bag") - head.dense_logits(h, table)).abs().max().item()
    assert err < FP64_TOL


def test_unigram_bias_path_is_exact():
    _, packed, head, table = make(unigram_bias=True)
    with torch.no_grad():
        head.unigram.normal_(0, 0.5)
    h = torch.randn(6, 48, dtype=torch.float64)
    err = (head(h, impl="bag") - head.dense_logits(h, table)).abs().max().item()
    assert err < FP64_TOL


def test_batched_shape_is_preserved():
    _, packed, head, _ = make()
    h = torch.randn(3, 5, 48, dtype=torch.float64)
    assert head(h).shape == (3, 5, packed.V)


# -------------------------------------------------------------- the C1 claim


def test_head_parameters_are_invariant_to_vocab_size():
    """C1: zero vocabulary-dependent parameters."""
    codec = KroneckerCodec(d_c=32, L_max=32, seed=0)
    counts = set()
    for V in (256, 4096, 50_000):
        packed = toy_vocab(V=V, L_max=32)
        head = KASHead(48, codec, packed, dtype=torch.float32)
        counts.add(head.head_parameters())
    assert len(counts) == 1, counts


def test_unigram_bias_is_the_only_vocab_dependent_term():
    codec = KroneckerCodec(d_c=32, L_max=32, seed=0)
    a = KASHead(48, codec, toy_vocab(V=1000, L_max=32), unigram_bias=True)
    b = KASHead(48, codec, toy_vocab(V=2000, L_max=32), unigram_bias=True)
    assert b.head_parameters() - a.head_parameters() == 1000


def test_occupancy_has_no_parameters():
    packed = toy_vocab()
    occ = Occupancy(packed)
    assert list(occ.parameters()) == []


def test_occupancy_rows_are_unit_norm():
    """The E4 diagnosis in one assertion: the output matrix has no per-token
    magnitude at all, so it structurally cannot hold a frequency prior."""
    packed = toy_vocab()
    dense = Occupancy(packed, dtype=torch.float64).to_dense_matrix()
    norms = dense.norm(dim=1)
    assert (norms - 1.0).abs().max().item() < 1e-14


def test_occupancy_matrix_matches_embedding_bag_path():
    packed = toy_vocab()
    occ = Occupancy(packed, dtype=torch.float64)
    dense = occ.to_dense_matrix()
    x = torch.randn(occ.n_cells, 9, dtype=torch.float64)
    assert (occ.matmul(x) - dense @ x).abs().max().item() < FP64_TOL


def test_occupancy_is_sparse():
    packed = toy_vocab(V=4096, L_max=64)
    occ = Occupancy(packed)
    assert occ.density < 0.01
