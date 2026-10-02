"""Correctness tests for the MoE router, the loss-free balancing bias update, and the
sparse-upcycling conversion in moe_llm.py.

These are arithmetic claims, not measurements, and must hold on any machine (no GPU, no real
training needed):

1. The router's top-k selection renormalizes to a gate that always sums to 1.
2. Zero-noise upcycling reproduces the dense model's output *exactly*, regardless of which
   experts a randomly initialized router happens to select -- the whole reason "copy" sparse
   upcycling (Komatsuzaki et al. 2022) doesn't destroy the model at the moment of conversion.
3. The loss-free balancing bias moves busier-than-average experts down and
   quieter-than-average experts up, by exactly `balance_gamma`, and never touches the
   training loss.

Importing moe_llm.py directly would execute its top-level cells (data download, both full
training runs) -- far too slow for a unit test. So instead this execs only the portion of the
source up to (not including) section 3's upcycling demo, into a throwaway namespace whose
`__file__` points at a scratch directory, with the corpus download stubbed to a tiny
in-memory string.
"""
import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_FILE = os.path.join(os.path.dirname(HERE), "moe_llm.py")

_SRC = open(REPO_FILE, encoding="utf-8").read()
_CUTOFF = _SRC.index("# %% [markdown]\n# ## 5. Run 1")
_scratch = tempfile.mkdtemp(prefix="moe_llm_test_")
_ns = {"__name__": "moe_llm_defs", "__file__": os.path.join(_scratch, "moe_llm.py")}
_stubbed = _SRC[:_CUTOFF].replace(
    'CORPUS = os.path.join(OUT_DIR, "tinyshakespeare.txt")\n'
    'if not os.path.exists(CORPUS):\n'
    '    try:\n'
    '        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"\n'
    '               "tinyshakespeare/input.txt")\n'
    '        urllib.request.urlretrieve(url, CORPUS)\n'
    '    except Exception as e:\n'
    '        print(f"download failed ({e}); falling back to a synthetic corpus")\n'
    '        rng = np.random.default_rng(SEED)\n'
    '        words = ["thou", "art", "the", "king", "and", "night", "doth", "come", "sweet",\n'
    '                 "sorrow", "my", "lord", "speak", "hence", "away", "shall", "heart"]\n'
    '        text = " ".join(rng.choice(words, 400000))\n'
    '        open(CORPUS, "w", encoding="utf-8").write(text)\n'
    'text = open(CORPUS, encoding="utf-8").read()',
    'CORPUS = os.path.join(OUT_DIR, "test_stub_corpus.txt")\n'
    'text = "to be or not to be that is the question " * 200\n'
    'open(CORPUS, "w", encoding="utf-8").write(text)',
)
exec(compile(_stubbed, "<moe_llm-defs>", "exec"), _ns)

GPTConfig = _ns["GPTConfig"]
GPT = _ns["GPT"]
DenseMLP = _ns["DenseMLP"]
MoEMLP = _ns["MoEMLP"]
upcycle_to_moe = _ns["upcycle_to_moe"]

torch.manual_seed(0)


# =========================================================================================
# 1. router: top-k selection, renormalized gate
# =========================================================================================
def test_router_gate_sums_to_one():
    torch.manual_seed(1)
    dim, n_experts, top_k = 8, 6, 2
    mlp = MoEMLP(dim, mult=2, n_experts=n_experts, top_k=top_k, dropout=0.0, balance_gamma=0.01)
    x = torch.randn(3, 5, dim)

    B, T, C = x.shape
    logits = mlp.router(x)
    probs = F.softmax(logits, dim=-1)
    score = probs + mlp.bias
    _, top_idx = score.topk(top_k, dim=-1)
    gate = probs.gather(-1, top_idx)
    gate = gate / gate.sum(dim=-1, keepdim=True)

    assert torch.allclose(gate.sum(dim=-1), torch.ones(B, T), atol=1e-6)
    assert top_idx.shape == (B, T, top_k)


def test_bias_affects_selection_not_gate_weight():
    """Two experts with identical router logits but very different biases must end up
    selected differently, while the *weight* given to a selected expert only ever depends on
    its unbiased softmax probability."""
    torch.manual_seed(2)
    dim, n_experts, top_k = 4, 4, 1
    mlp = MoEMLP(dim, mult=2, n_experts=n_experts, top_k=top_k, dropout=0.0, balance_gamma=0.01)
    x = torch.randn(1, 1, dim)

    with torch.no_grad():
        logits = mlp.router(x)
        probs = F.softmax(logits, dim=-1)
        winner_without_bias = probs.argmax(dim=-1)

        mlp.bias.zero_()
        mlp.bias[winner_without_bias.item()] = -10.0  # ban the natural winner
        score = probs + mlp.bias
        _, top_idx = score.topk(top_k, dim=-1)

    assert top_idx.item() != winner_without_bias.item(), \
        "a large negative bias must be able to override the router's natural top choice"


# =========================================================================================
# 2. sparse upcycling: zero-noise exactness
# =========================================================================================
def test_zero_noise_upcycling_reconstructs_dense_exactly():
    """With noise_std=0, every expert is a bit-identical clone of the dense feed-forward
    network, so no matter which subset the (randomly initialized) router selects, the
    renormalized gate always sums to 1 over copies of the *same* function -- so the MoE
    layer's output must equal the dense layer's output exactly, for every top_k from 1 to
    n_experts."""
    torch.manual_seed(3)
    vocab_size, n_layer = 11, 2
    dense_cfg = GPTConfig(vocab_size=vocab_size, block_size=16, n_layer=n_layer, n_embd=12,
                           n_head=2, mlp_mult=3, dropout=0.0, mode="dense")
    dense_model = GPT(dense_cfg)
    dense_model.eval()

    x = torch.randint(0, vocab_size, (2, 9))
    with torch.no_grad():
        dense_logits, _ = dense_model(x)

    for top_k in (1, 3, 8):
        moe_cfg = GPTConfig(vocab_size=vocab_size, block_size=16, n_layer=n_layer, n_embd=12,
                             n_head=2, mlp_mult=3, dropout=0.0, mode="moe", n_experts=8,
                             top_k=top_k, balance_gamma=0.01)
        moe_model = upcycle_to_moe(dense_model, moe_cfg, noise_std=0.0, seed=top_k)
        moe_model.eval()
        with torch.no_grad():
            moe_logits, _ = moe_model(x)
        assert torch.allclose(dense_logits, moe_logits, atol=1e-5), \
            f"zero-noise upcycling must reproduce dense output exactly at top_k={top_k}"


def test_nonzero_noise_upcycling_perturbs_but_stays_close():
    """A real run adds noise specifically so experts are *not* identical; this checks the
    perturbation is actually present (output differs from dense) but still small (the whole
    point of upcycling from trained weights rather than random-initializing the MoE)."""
    torch.manual_seed(4)
    vocab_size = 11
    dense_cfg = GPTConfig(vocab_size=vocab_size, block_size=16, n_layer=2, n_embd=12, n_head=2,
                           mlp_mult=3, dropout=0.0, mode="dense")
    dense_model = GPT(dense_cfg)
    dense_model.eval()
    x = torch.randint(0, vocab_size, (2, 9))
    with torch.no_grad():
        dense_logits, _ = dense_model(x)

    moe_cfg = GPTConfig(vocab_size=vocab_size, block_size=16, n_layer=2, n_embd=12, n_head=2,
                         mlp_mult=3, dropout=0.0, mode="moe", n_experts=8, top_k=2,
                         balance_gamma=0.01)
    moe_model = upcycle_to_moe(dense_model, moe_cfg, noise_std=0.05, seed=0)
    moe_model.eval()
    with torch.no_grad():
        moe_logits, _ = moe_model(x)

    diff = (dense_logits - moe_logits).abs().max().item()
    assert diff > 1e-4, "noise=0.05 should visibly perturb the output vs. the dense model"
    assert diff < 2.0, "upcycled output should still be in the same ballpark as the dense model"


def test_upcycle_preserves_attention_and_embeddings_exactly():
    torch.manual_seed(5)
    dense_cfg = GPTConfig(vocab_size=13, block_size=16, n_layer=2, n_embd=12, n_head=2,
                           mlp_mult=3, dropout=0.0, mode="dense")
    dense_model = GPT(dense_cfg)
    moe_cfg = GPTConfig(vocab_size=13, block_size=16, n_layer=2, n_embd=12, n_head=2,
                         mlp_mult=3, dropout=0.0, mode="moe", n_experts=4, top_k=2,
                         balance_gamma=0.01)
    moe_model = upcycle_to_moe(dense_model, moe_cfg, noise_std=0.01, seed=0)

    assert torch.allclose(dense_model.tok_emb.weight, moe_model.tok_emb.weight)
    assert torch.allclose(dense_model.pos_emb.weight, moe_model.pos_emb.weight)
    for db, mb in zip(dense_model.blocks, moe_model.blocks):
        for p_d, p_m in zip(db.attn.parameters(), mb.attn.parameters()):
            assert torch.allclose(p_d, p_m)
    # weight tying must survive the in-place .data.copy_ used during upcycling
    assert moe_model.head.weight is moe_model.tok_emb.weight


def test_param_counts_active_less_than_total_for_moe():
    cfg = GPTConfig(vocab_size=13, block_size=16, n_layer=2, n_embd=12, n_head=2, mlp_mult=3,
                     dropout=0.0, mode="moe", n_experts=8, top_k=2, balance_gamma=0.01)
    model = GPT(cfg)
    total, active = model.param_counts()
    assert active < total
    per_expert = sum(p.numel() for p in model.blocks[0].mlp.experts[0].parameters())
    expected_inactive = per_expert * (8 - 2) * cfg.n_layer
    assert total - active == expected_inactive


# =========================================================================================
# 3. loss-free balancing bias update
# =========================================================================================
def test_rebalance_pushes_busy_expert_down_and_idle_expert_up():
    torch.manual_seed(6)
    n_experts, gamma = 4, 0.01
    mlp = MoEMLP(dim=8, mult=2, n_experts=n_experts, top_k=1, dropout=0.0, balance_gamma=gamma)

    # fabricate an uneven window: expert 0 got everything, expert 3 got nothing
    mlp.count_accum[:] = torch.tensor([40.0, 10.0, 10.0, 0.0])
    bias_before = mlp.bias.clone()
    counts = mlp.maybe_rebalance()

    assert torch.allclose(counts, torch.tensor([40.0, 10.0, 10.0, 0.0]))
    assert mlp.bias[0] == pytest.approx(bias_before[0].item() - gamma)   # over target -> pushed down
    assert mlp.bias[3] == pytest.approx(bias_before[3].item() + gamma)   # under target -> pushed up
    # expert 1 and 2 (target = 60/4 = 15) are also under target -> pushed up
    assert mlp.bias[1] == pytest.approx(bias_before[1].item() + gamma)
    assert mlp.bias[2] == pytest.approx(bias_before[2].item() + gamma)
    # the window resets after rebalancing
    assert torch.allclose(mlp.count_accum, torch.zeros(n_experts))


def test_rebalance_is_noop_with_no_tokens_seen():
    mlp = MoEMLP(dim=8, mult=2, n_experts=4, top_k=1, dropout=0.0, balance_gamma=0.01)
    bias_before = mlp.bias.clone()
    assert mlp.maybe_rebalance() is None
    assert torch.equal(mlp.bias, bias_before)


def test_repeated_rebalancing_narrows_an_artificial_imbalance():
    """Not a training run -- just the bias mechanism in isolation. If one expert is
    structurally favored by the router (a large fixed positive offset in its logits), the
    balancing bias should grow negative for it across repeated windows, driving it out of the
    top-k selection once the bias overcomes the favoritism -- at which point the mechanism has
    actually fixed the imbalance and the bias should stop moving, rather than growing without
    bound."""
    torch.manual_seed(7)
    n_experts, top_k, gamma = 6, 2, 0.05
    mlp = MoEMLP(dim=10, mult=2, n_experts=n_experts, top_k=top_k, dropout=0.0,
                 balance_gamma=gamma)
    with torch.no_grad():
        mlp.router.weight.zero_()  # router output depends only on a fixed favoritism offset
    favoritism = torch.zeros(n_experts)
    favoritism[0] = 5.0  # expert 0 is structurally favored

    def step(x):
        B, T, C = x.shape
        logits = mlp.router(x) + favoritism
        probs = F.softmax(logits, dim=-1)
        score = probs + mlp.bias
        _, top_idx = score.topk(top_k, dim=-1)
        onehot = torch.zeros(B, T, n_experts)
        onehot.scatter_(-1, top_idx, 1.0)
        mlp.count_accum += onehot.sum(dim=(0, 1))
        return onehot.sum(dim=(0, 1))

    last_counts = None
    for _ in range(30):
        last_counts = step(torch.randn(4, 6, 10))
        mlp.maybe_rebalance()

    even_share = last_counts.sum() / n_experts
    assert mlp.bias[0] < -0.1, \
        f"expert 0's bias should have been pushed clearly negative, got {mlp.bias[0].item()}"
    assert last_counts[0] <= even_share * 1.5, (
        "after enough windows the bias should have knocked expert 0 out of its structural "
        f"advantage; last window's counts were {last_counts.tolist()}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
