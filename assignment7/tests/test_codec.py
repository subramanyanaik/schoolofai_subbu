import numpy as np
import pytest

from kronecker_v2.codec import KroneckerCodec

MULTISCRIPT = [
    "apple",
    "counterrevolutionaries",
    "भारत",           # Devanagari
    "తెలుగు",  # Telugu
    "中文测试",           # CJK
    "Кириллица",  # Cyrillic
    "עברית",     # Hebrew
    "العربية",  # Arabic
    "\U0001f37a",                          # emoji
]


@pytest.mark.parametrize("d_c", [16, 32, 64])
def test_roundtrip_random_bytes(d_c):
    codec = KroneckerCodec(d_c=d_c, L_max=64)
    rng = np.random.default_rng(0)
    seqs = [
        bytes(rng.integers(0, 256, size=int(n), dtype=np.uint8).tolist())
        for n in rng.integers(1, 65, size=2000)
    ]
    assert codec.roundtrip(seqs) == seqs


def test_roundtrip_multiscript():
    codec = KroneckerCodec(d_c=32, L_max=128)
    seqs = [s.encode("utf-8") for s in MULTISCRIPT]
    assert codec.roundtrip(seqs) == seqs


def test_sign_and_nearest_neighbour_decodes_agree():
    codec = KroneckerCodec(d_c=32, L_max=64)
    rng = np.random.default_rng(1)
    seqs = [
        bytes(rng.integers(0, 256, size=int(n), dtype=np.uint8).tolist())
        for n in rng.integers(1, 65, size=500)
    ]
    K = codec.encode(seqs)
    assert codec.decode(K, method="sign") == codec.decode(K, method="nn")


def test_inversion_is_exact_not_approximate():
    """M_pos = (M_pos @ Phi) @ Phi.T to machine precision, no pseudo-inverse."""
    codec = KroneckerCodec(d_c=32, L_max=64)
    seqs = [b"hello", b"a", b"x" * 64]
    M = codec.encode_positions(seqs)
    K = codec.encode(seqs)
    back = codec.decode_positions(K)
    assert np.abs(M - back).max() < 1e-14


def test_rows_are_unit_norm():
    """Every codec vector has norm 1 -- this is what makes the KAS output
    matrix magnitude-free, which is E4's whole diagnosis."""
    codec = KroneckerCodec(d_c=32, L_max=64)
    seqs = [b"a", b"ab", b"abc", b"x" * 64, b"y" * 100]
    K = codec.encode(seqs)
    np.testing.assert_allclose(np.linalg.norm(K, axis=1), 1.0, atol=1e-13)


def test_length_falls_out_of_the_decode():
    codec = KroneckerCodec(d_c=32, L_max=64)
    for n in range(1, 33):
        s = bytes(range(1, n + 1))
        assert codec.roundtrip([s])[0] == s


def test_zero_byte_is_representable():
    """b'\\x00' is a real byte value, not padding: the absent marker is the
    zero *vector*, not the zero byte."""
    codec = KroneckerCodec(d_c=32, L_max=16)
    seqs = [b"\x00", b"\x00\x00\x00", b"a\x00b"]
    assert codec.roundtrip(seqs) == seqs


def test_v1_variant_matches_published_codec():
    codec = KroneckerCodec(variant="v1", L_max=16)
    assert (codec.d_c, codec.L_max, codec.D) == (256, 16, 4096)
    assert np.array_equal(codec.G, np.eye(256))
    assert np.array_equal(codec.Phi, np.eye(16))
    assert codec.roundtrip([b"apple"]) == [b"apple"]


def test_v1_truncates_where_v2_does_not_at_matched_D():
    long = b"counterrevolutionaries"  # 22 bytes
    v1 = KroneckerCodec(variant="v1", L_max=16)          # D = 4096
    v2 = KroneckerCodec(d_c=32, L_max=128)               # D = 4096
    assert v1.D == v2.D == 4096
    assert v1.roundtrip([long])[0] != long
    assert v2.roundtrip([long])[0] == long


def test_truncate_beats_alias_on_overflow():
    """The design claim that lost: aliasing corrupts every position at once,
    truncation keeps a correct prefix."""
    L_max = 64
    rng = np.random.default_rng(2)
    seqs = [bytes(rng.integers(0, 256, size=96, dtype=np.uint8).tolist()) for _ in range(64)]
    trunc = KroneckerCodec(d_c=32, L_max=L_max, overflow="truncate")
    alias = KroneckerCodec(d_c=32, L_max=L_max, overflow="alias")
    a_t = trunc.roundtrip_stats(seqs)["byte_accuracy"]
    a_a = alias.roundtrip_stats(seqs)["byte_accuracy"]
    assert a_t == pytest.approx(64 / 96, abs=1e-9)
    assert a_a < a_t


def test_deterministic_forward_and_reverse():
    """Problem 5, stated as a test."""
    codec = KroneckerCodec(d_c=32, L_max=64)
    k1 = codec.encode([b"apple"])
    k2 = codec.encode([b"apple"])
    assert np.array_equal(k1, k2)                       # forward deterministic
    assert codec.decode(k1) == codec.decode(k2) == [b"apple"]  # reverse too
