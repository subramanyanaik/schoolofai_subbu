"""Kronecker Embeddings V2 -- Additive Softmax over Surface Forms (KAS).

Assignment 7, problem #5: make the forward-deterministic Kronecker codec
reverse-deterministic, and use that to delete the output head.
"""

from .codebooks import (
    BIT_COLUMNS,
    byte_codebook,
    coherence_stats,
    position_basis,
    sylvester_hadamard,
    welch_bound,
)
from .codec import KroneckerCodec

__all__ = [
    "BIT_COLUMNS",
    "KroneckerCodec",
    "byte_codebook",
    "coherence_stats",
    "position_basis",
    "sylvester_hadamard",
    "welch_bound",
]
