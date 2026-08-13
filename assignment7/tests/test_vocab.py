import numpy as np
import pytest

from kronecker_v2.kas_head import Occupancy
from kronecker_v2.vocab import PackedVocab, build_large_vocab


def toy(V=400, L_max=16, seed=0, hi=40):
    rng = np.random.default_rng(seed)
    return [
        bytes(rng.integers(0, 256, size=int(n), dtype=np.uint8).tolist())
        for n in rng.integers(1, hi, size=V)
    ], L_max


def test_packing_is_length_sorted():
    toks, L = toy()
    p = PackedVocab(toks, L)
    assert np.all(np.diff(p.true_lengths) >= 0)
    assert np.all(np.diff(p.lengths) >= 0)


def test_order_roundtrips():
    toks, L = toy()
    p = PackedVocab(toks, L)
    ids = np.arange(p.V)
    assert np.array_equal(p.to_packed(p.to_original(ids)), ids)
    for i in range(0, p.V, 37):
        assert p.token_bytes[i] == toks[p.order[i]]


def test_hi_gives_the_contiguous_suffix():
    """The property the whole head design rests on."""
    toks, L = toy()
    p = PackedVocab(toks, L)
    for pos in range(L):
        lo = p.hi[pos]
        assert all(len(t) > pos for t in p._sorted_bytes[lo:])
        assert all(len(t) <= pos for t in p._sorted_bytes[:lo])


def test_packing_order_is_independent_of_L_max():
    """E5 sweeps L_max; the tokenized data bins must stay valid."""
    toks, _ = toy(hi=200)
    a = PackedVocab(toks, 32)
    b = PackedVocab(toks, 128)
    assert np.array_equal(a.order, b.order)
    assert np.array_equal(a.inv_order, b.inv_order)
    assert not np.array_equal(a.lengths, b.lengths)  # clipping does differ


def test_flat_arrays_are_consistent():
    toks, L = toy()
    p = PackedVocab(toks, L)
    assert p.flat_bytes.size == p.total_bytes == int(p.lengths.sum())
    for i in range(0, p.V, 53):
        lo = p.offsets[i]
        seg = p.flat_bytes[lo : lo + p.lengths[i]]
        assert bytes(seg.tolist()) == p._sorted_bytes[i]


def test_truncation_is_reported_not_hidden():
    toks, _ = toy(hi=200)
    p = PackedVocab(toks, 32)
    assert p.n_truncated == int((p.true_lengths > 32).sum())
    assert p.summary()["frac_truncated"] == pytest.approx(p.n_truncated / p.V)


def test_occupancy_matches_the_packing():
    toks, L = toy()
    p = PackedVocab(toks, L)
    occ = Occupancy(p, dtype=None or __import__("torch").float64)
    dense = occ.to_dense_matrix().numpy()
    for i in range(0, p.V, 41):
        row = dense[i]
        nz = np.flatnonzero(row)
        assert nz.size == p.lengths[i]
        for pos, b in enumerate(p._sorted_bytes[i]):
            assert row[b * L + pos] == pytest.approx(1.0 / np.sqrt(p.lengths[i]))


def test_build_large_vocab_reaches_target():
    base = [bytes([i]) for i in range(64)]
    texts = ["the quick brown fox jumps over the lazy dog " * 20]
    big = build_large_vocab(500, base, texts, L_max=32, seed=0)
    assert len(big) == 500
    assert len(set(big)) == 500
    assert big[:64] == base
    stats = build_large_vocab.last_stats
    assert stats["harvested"] > 0


def test_gpt2_vocab_loads():
    from kronecker_v2.vocab import gpt2_token_bytes

    toks = gpt2_token_bytes()
    assert len(toks) == 50257
    assert all(len(t) >= 1 for t in toks)
    p = PackedVocab(toks, 128)
    assert p.n_truncated == 0
    assert p.max_len <= 128
