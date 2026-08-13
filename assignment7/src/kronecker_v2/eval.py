"""Evaluation: bits-per-byte, perplexity, the unigram floor, token buckets.

bits-per-byte rather than bits-per-token, because the arms are compared across
tokenizations and scripts.  bpb = total_nll_nats / (ln2 * total_target_bytes),
where a target token's byte count is its *true* (unclipped) surface length.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


def _autocast(device: str, enabled: bool = True):
    if device == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def evaluate(
    model,
    stream,
    device: str,
    byte_lengths: np.ndarray,
    batch_size: int = 8,
    max_batches: int | None = 200,
    amp: bool = True,
) -> dict:
    model.eval()
    blen = torch.from_numpy(np.asarray(byte_lengths, dtype=np.float64)).to(device)
    tot_nll, tot_tok, tot_bytes = 0.0, 0, 0.0
    for x, y in stream.sequential(batch_size, max_batches=max_batches):
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device)
        with _autocast(device, amp):
            logits, _ = model(xb)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            yb.reshape(-1),
            reduction="sum",
        )
        tot_nll += nll.item()
        tot_tok += yb.numel()
        tot_bytes += blen[yb.reshape(-1)].sum().item()
    model.train()
    nll_per_tok = tot_nll / max(tot_tok, 1)
    return {
        "nll_nats_per_token": nll_per_tok,
        "ppl": math.exp(min(nll_per_tok, 50.0)),
        "bpb": tot_nll / (LN2 * max(tot_bytes, 1.0)),
        "n_tokens": tot_tok,
        "n_bytes": tot_bytes,
    }


def unigram_floor(
    train_ids: np.ndarray,
    val_stream,
    byte_lengths: np.ndarray,
    vocab_size: int,
    batch_size: int = 8,
    max_batches: int | None = 200,
) -> dict:
    """Bits-per-byte of a model that knows only token frequencies.

    Fitted on training counts (add-one smoothed), scored on the same validation
    windows the models see.  This is the floor KAS turns out to sit on.
    """
    counts = np.bincount(np.asarray(train_ids, dtype=np.int64), minlength=vocab_size)
    probs = (counts + 1.0) / (counts.sum() + vocab_size)
    logp = np.log(probs)
    tot_nll, tot_tok, tot_bytes = 0.0, 0, 0.0
    for _, y in val_stream.sequential(batch_size, max_batches=max_batches):
        flat = y.reshape(-1)
        tot_nll += float(-logp[flat].sum())
        tot_tok += flat.size
        tot_bytes += float(byte_lengths[flat].sum())
    return {
        "nll_nats_per_token": tot_nll / max(tot_tok, 1),
        "bpb": tot_nll / (LN2 * max(tot_bytes, 1.0)),
        "n_tokens": tot_tok,
    }


@torch.no_grad()
def per_token_nll(
    model,
    stream,
    device: str,
    vocab_size: int,
    batch_size: int = 8,
    max_batches: int | None = 200,
    amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Summed NLL and occurrence count per target token id."""
    model.eval()
    nll_sum = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    counts = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    for x, y in stream.sequential(batch_size, max_batches=max_batches):
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device).reshape(-1)
        with _autocast(device, amp):
            logits, _ = model(xb)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), yb, reduction="none"
        ).double()
        nll_sum.index_add_(0, yb, nll)
        counts.index_add_(0, yb, torch.ones_like(nll))
    model.train()
    return nll_sum.cpu().numpy(), counts.cpu().numpy()


@torch.no_grad()
def bpb_of_text(
    model,
    text: str,
    packed,
    device: str,
    block_size: int = 256,
    batch_size: int = 4,
    max_tokens: int = 200_000,
    amp: bool = True,
) -> dict:
    """Bits-per-byte on a raw string.  Used for the out-of-domain arms."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ids = np.asarray(enc.encode_ordinary(text)[:max_tokens], dtype=np.int64)
    ids = packed.inv_order[ids].astype(np.int64)
    n = (ids.size - 1) // block_size
    if n == 0:
        return {"bpb": float("nan"), "n_tokens": 0}

    blen = torch.from_numpy(packed.true_lengths.astype(np.float64)).to(device)
    model.eval()
    tot_nll, tot_tok, tot_bytes = 0.0, 0, 0.0
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        x = np.stack([ids[i * block_size : (i + 1) * block_size] for i in idx])
        y = np.stack([ids[i * block_size + 1 : (i + 1) * block_size + 1] for i in idx])
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device).reshape(-1)
        with _autocast(device, amp):
            logits, _ = model(xb)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), yb, reduction="sum"
        )
        tot_nll += nll.item()
        tot_tok += yb.numel()
        tot_bytes += blen[yb].sum().item()
    model.train()
    return {
        "bpb": tot_nll / (LN2 * max(tot_bytes, 1.0)),
        "nll_nats_per_token": tot_nll / max(tot_tok, 1),
        "n_tokens": tot_tok,
        "n_bytes": tot_bytes,
    }
