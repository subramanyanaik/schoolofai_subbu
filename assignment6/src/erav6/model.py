"""
model.py — a small causal transformer that actually honours the batch.

The model is the least interesting part of this assignment, and it is included
for one reason: the packing metadata has to be USED. A system that emits
beautiful segment ids and position ids and then feeds a plain causal mask to
the model has not demonstrated anything -- the masks would be decoration.

So this model:

  * builds its attention mask from `segment_ids`, which makes co-packed samples
    genuinely invisible to each other rather than nominally isolated;
  * indexes position embeddings with `position_ids`, so a sample packed second
    really does start at position 0;
  * returns PER-TOKEN cross-entropy, not a scalar, because the token-level
    perplexity trace needs the individual values and recovering them later
    would mean re-running the model at the same training state.

Padding is handled by the same mechanism as isolation: pad positions carry
segment id 0, which is never equal to any real segment id, so nothing attends
to them and they attend to nothing.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100
NEG_INF = -1e9


def resolve_device(requested: str = "auto") -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_attention_bias(segment_ids: torch.Tensor) -> torch.Tensor:
    """Additive attention bias from segment ids.

    allowed(i, j)  <=>  j <= i                     (causal)
                   and  seg[i] == seg[j]           (same sample)
                   and  seg[i] != 0                (not padding)

    Returns [B, 1, T, T] of 0.0 where allowed and a large negative where not.
    Padding rows are fully masked, which would make their softmax degenerate --
    those rows are never read, because every pad position is excluded from the
    loss by `loss_mask`. The diagonal is left open so the softmax stays finite.
    """
    b, t = segment_ids.shape
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=segment_ids.device))
    same = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)      # [B, T, T]
    real = (segment_ids != 0).unsqueeze(2) & (segment_ids != 0).unsqueeze(1)
    allowed = causal.unsqueeze(0) & same & real
    # keep the diagonal open so no row is entirely -inf
    eye = torch.eye(t, dtype=torch.bool, device=segment_ids.device).unsqueeze(0)
    allowed = allowed | eye
    return torch.where(allowed, 0.0, NEG_INF).unsqueeze(1)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must divide by n_head")
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att + bias                       # the packing mask enters here
        att = F.softmax(att, dim=-1)
        if self.dropout and self.training:
            att = F.dropout(att, p=self.dropout)
        out = (att @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), bias)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int,
                 d_ff: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight        # tied
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                segment_ids: torch.Tensor) -> torch.Tensor:
        bias = build_attention_bias(segment_ids)
        x = self.tok(input_ids) + self.pos(position_ids.clamp(max=self.max_seq_len - 1))
        for blk in self.blocks:
            x = blk(x, bias)
        return self.head(self.ln_f(x))

    def loss(self, input_ids: torch.Tensor, labels: torch.Tensor,
             loss_mask: torch.Tensor, position_ids: torch.Tensor,
             segment_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean loss over loss-bearing tokens, per-token loss [B, T]).

        The mean is over the number of LOSS-BEARING tokens, not over token
        positions. Dividing by positions would make a batch's gradient depend on
        how much padding it happened to carry, so an efficiently packed batch
        would be weighted differently from a sparse one -- a silent coupling
        between packing efficiency and effective learning rate.
        """
        logits = self.forward(input_ids, position_ids, segment_ids)
        safe_labels = labels.clamp(min=0)
        per_token = F.cross_entropy(
            logits.view(-1, self.vocab_size), safe_labels.view(-1),
            reduction="none").view(labels.shape)
        mask = loss_mask.to(per_token.dtype)
        per_token = per_token * mask
        denom = mask.sum().clamp(min=1.0)
        return per_token.sum() / denom, per_token

    @staticmethod
    def stack(sequences, device: torch.device) -> dict:
        """Turn a list of PackedSequence into the five aligned tensors.

        Kept here rather than in packing.py so that packing stays a pure,
        torch-free description of the batch -- the tests can check mask
        correctness without importing a deep-learning framework.
        """
        return dict(
            input_ids=torch.tensor([s.input_ids for s in sequences],
                                   dtype=torch.long, device=device),
            labels=torch.tensor([s.labels for s in sequences],
                                dtype=torch.long, device=device),
            loss_mask=torch.tensor([s.loss_mask for s in sequences],
                                   dtype=torch.long, device=device),
            position_ids=torch.tensor([s.position_ids for s in sequences],
                                      dtype=torch.long, device=device),
            segment_ids=torch.tensor([s.segment_ids for s in sequences],
                                     dtype=torch.long, device=device),
        )

    @torch.no_grad()
    def eval_loss(self, input_ids: torch.Tensor, labels: torch.Tensor,
                  loss_mask: torch.Tensor, position_ids: torch.Tensor,
                  segment_ids: torch.Tensor) -> float:
        was_training = self.training
        self.eval()
        mean, _ = self.loss(input_ids, labels, loss_mask, position_ids, segment_ids)
        if was_training:
            self.train()
        return float(mean.item())
