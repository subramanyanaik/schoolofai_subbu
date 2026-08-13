import numpy as np
import pytest
import torch

from kronecker_v2.codec import KroneckerCodec
from kronecker_v2.model import ARMS, GPT, KroneckerEmbedding
from kronecker_v2.train import build_arm
from kronecker_v2.vocab import PackedVocab

SMALL = dict(d_model=32, n_layer=2, n_head=2, block_size=16)


def toy_tokens(V=300, seed=0, hi=40):
    rng = np.random.default_rng(seed)
    return [
        bytes(rng.integers(0, 256, size=int(n), dtype=np.uint8).tolist())
        for n in rng.integers(1, hi, size=V)
    ]


# ----------------------------------------------- the dual of the E2 gate


@pytest.mark.parametrize("d_c,L_max", [(16, 32), (32, 64)])
def test_table_equals_direct_encode(d_c, L_max):
    """KroneckerEmbedding.table() == codec.encode(bytes) @ W_proj, exactly.

    This is what licenses never materialising the |V| x D codec table.
    """
    codec = KroneckerCodec(d_c=d_c, L_max=L_max, seed=0)
    packed = PackedVocab(toy_tokens(hi=L_max), L_max)
    emb = KroneckerEmbedding(codec, packed, 24, dtype=torch.float64)
    K = torch.from_numpy(codec.encode(packed.token_bytes))
    ref = K @ emb.W_proj
    assert (emb.table() - ref).abs().max().item() < 1e-11


def test_table_equals_direct_encode_with_znorm():
    """The V1 arm z-normalises the codec vector (arXiv:2605.29459 sec. 3)."""
    codec = KroneckerCodec(variant="v1", L_max=16)
    packed = PackedVocab(toy_tokens(hi=16), 16)
    emb = KroneckerEmbedding(codec, packed, 24, znorm=True, dtype=torch.float64)
    K = torch.from_numpy(codec.encode(packed.token_bytes))
    Kz = (K - K.mean(dim=1, keepdim=True)) / K.std(dim=1, unbiased=False, keepdim=True)
    ref = Kz @ emb.W_proj
    assert (emb.table() - ref).abs().max().item() < 1e-9


# ---------------------------------------------------------------- the arms


@pytest.mark.parametrize("arm", ARMS)
def test_arm_forward_and_backward(arm):
    model, packed, _ = build_arm(arm, toy_tokens(), d_c=16, L_max=32, **SMALL)
    idx = torch.randint(0, packed.V, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, packed.V)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("arm", ARMS)
def test_no_phantom_parameters(arm):
    """Trap 5: a stray nn.Module assignment registers parameters that never
    get a gradient.  Training is unaffected -- only the reported parameter
    count is wrong, which is the number this project is about."""
    model, _, _ = build_arm(arm, toy_tokens(), d_c=16, L_max=32, **SMALL)
    assert model.unused_parameters() == []


def test_occupancy_is_shared_not_duplicated():
    model, _, _ = build_arm("a2_kronf_kas", toy_tokens(), d_c=16, L_max=32, **SMALL)
    assert model.wte.occ is model.lm_head.occ is model.occ
    assert list(model.occ.parameters()) == []


def test_tying_removes_a_whole_projection():
    toks = toy_tokens()
    untied, _, _ = build_arm("a2_kronf_kas", toks, d_c=16, L_max=32, **SMALL)
    tied, _, codec = build_arm("a3_kronf_kas_tied", toks, d_c=16, L_max=32, **SMALL)
    assert tied.lm_head.is_tied and tied.lm_head.W_out is None
    assert untied.n_params() - tied.n_params() == codec.D * SMALL["d_model"]


def test_unigram_arm_adds_exactly_one_scalar_per_token():
    toks = toy_tokens()
    base, packed, _ = build_arm("a2_kronf_kas", toks, d_c=16, L_max=32, **SMALL)
    uni, _, _ = build_arm("a4_kas_unigram", toks, d_c=16, L_max=32, **SMALL)
    assert uni.n_params() - base.n_params() == packed.V


def test_kas_head_params_do_not_grow_with_vocab():
    """C1, at the level of the whole model rather than the head."""
    heads = set()
    for V in (300, 3000):
        model, _, _ = build_arm("a2_kronf_kas", toy_tokens(V=V), d_c=16, L_max=32, **SMALL)
        heads.add(model.param_breakdown()["head"])
    assert len(heads) == 1, heads


def test_dense_head_does_grow_with_vocab():
    """The control.  a0 ties its head to the embedding, so the vocabulary cost
    is booked under the input table -- but it is still |V| * d_model, and it
    still grows.  a2's total does not move at all."""
    a0 = [
        build_arm("a0_bpe_tied", toy_tokens(V=V), **SMALL)[0].n_params()
        for V in (300, 3000)
    ]
    a2 = [
        build_arm("a2_kronf_kas", toy_tokens(V=V), d_c=16, L_max=32, **SMALL)[0].n_params()
        for V in (300, 3000)
    ]
    assert a0[1] - a0[0] == 2700 * SMALL["d_model"]
    assert a2[1] - a2[0] == 0


def test_a1_uses_the_published_v1_codec():
    _, _, codec = build_arm("a1_v1_dense", toy_tokens(), **SMALL)
    assert (codec.variant, codec.d_c, codec.L_max, codec.D) == ("v1", 256, 16, 4096)


def test_model_is_deterministic_given_a_seed():
    toks = toy_tokens()
    a, _, _ = build_arm("a2_kronf_kas", toks, d_c=16, L_max=32, seed=3, **SMALL)
    b, _, _ = build_arm("a2_kronf_kas", toks, d_c=16, L_max=32, seed=3, **SMALL)
    idx = torch.randint(0, 300, (2, 16))
    assert torch.equal(a(idx)[0], b(idx)[0])
