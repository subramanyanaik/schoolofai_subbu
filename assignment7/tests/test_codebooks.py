import numpy as np
import pytest

from kronecker_v2.codebooks import (
    BIT_COLUMNS,
    byte_codebook,
    coherence_stats,
    position_basis,
    sylvester_hadamard,
    welch_bound,
)


def test_hadamard_is_orthogonal():
    H = sylvester_hadamard(256)
    assert H.dtype == np.float64
    assert np.array_equal(H @ H.T, 256 * np.eye(256))


def test_hadamard_bit_columns_encode_bits():
    H = sylvester_hadamard(256)
    for k, col in enumerate(BIT_COLUMNS):
        bit = (np.arange(256) >> k) & 1
        assert np.array_equal(H[:, col], np.where(bit, -1.0, 1.0))


@pytest.mark.parametrize("d_c", [8, 16, 32, 64, 128])
def test_byte_codebook_rows_distinct_and_unit_norm(d_c):
    G = byte_codebook(d_c, seed=0)
    assert G.shape == (256, d_c)
    assert G.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(G, axis=1), 1.0, atol=1e-15)
    stats = coherence_stats(G)
    assert stats["rows_distinct"], "bit-columns must keep all 256 rows distinct"


@pytest.mark.parametrize("d_c", [16, 32, 64])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_sign_read_recovers_the_byte(d_c, seed):
    """The 8 forced bit-columns make decoding an O(8) sign read."""
    G = byte_codebook(d_c, seed=seed)
    bits = (G[:, :8] < 0).astype(np.int64)
    recovered = (bits * (1 << np.arange(8))).sum(axis=1)
    assert np.array_equal(recovered, np.arange(256))


def test_onehot_is_v1():
    G = byte_codebook(256, method="onehot")
    assert np.array_equal(G, np.eye(256))


@pytest.mark.parametrize("L", [1, 2, 8, 16, 17, 32, 33, 64, 128])
def test_position_basis_is_orthogonal(L):
    Phi = position_basis(L, "dft")
    assert Phi.shape == (L, L)
    np.testing.assert_allclose(Phi @ Phi.T, np.eye(L), atol=1e-13)
    np.testing.assert_allclose(Phi.T @ Phi, np.eye(L), atol=1e-13)


def test_identity_position_basis_is_v1():
    assert np.array_equal(position_basis(16, "identity"), np.eye(16))


def test_coherence_respects_welch_bound():
    for d_c in (16, 32, 64, 128):
        stats = coherence_stats(byte_codebook(d_c, seed=0))
        assert stats["max_coherence"] >= stats["welch_bound"] - 1e-12
        assert welch_bound(256, d_c) > 0


def test_codebooks_are_deterministic():
    a = byte_codebook(32, seed=3)
    b = byte_codebook(32, seed=3)
    assert np.array_equal(a, b)
    c = byte_codebook(32, seed=4)
    assert not np.array_equal(a, c)
