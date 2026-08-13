"""Corpus -> uint32 token bins, in packed-vocabulary order.

Token ids are stored *already remapped* into ``PackedVocab`` order, so the
bins are valid for every codec configuration: the packing sorts by true byte
length, which does not depend on L_max.  E5 sweeps L_max and must not
re-tokenize.

Trap 9: Windows stdout is cp1252.  Every file read/write here passes
``encoding="utf-8"`` explicitly, and scripts call ``force_utf8_stdout()``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

N_VAL_DOCS = 4000
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def _maybe_truststore() -> None:
    """Only needed behind a TLS-intercepting corporate proxy.

    It makes huggingface_hub validate against the OS certificate store instead
    of certifi's bundle.  It does **not** disable verification.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


def iter_documents(source: str, limit: int | None = None):
    """Yield raw document strings from ``source``."""
    if source == "fineweb":
        _maybe_truststore()
        import datasets

        ds = datasets.load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                return
            yield row["text"]
        return

    if source == "local_parquet":
        import pyarrow.parquet as pq

        path = Path(".hfhome/ccnews/plain_text/train-00000-of-00005.parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Use --source fineweb on a fresh machine."
            )
        table = pq.read_table(path, columns=["text"])
        for i, val in enumerate(table.column("text")):
            if limit is not None and i >= limit:
                return
            yield val.as_py()
        return

    path = Path(source)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        for i, doc in enumerate(text.split("\n\n")):
            if limit is not None and i >= limit:
                return
            if doc.strip():
                yield doc
        return

    raise ValueError(f"unknown corpus source {source!r}")


def build_bins(
    source: str,
    packed,
    target_tokens: int,
    data_dir: Path = DEFAULT_DATA_DIR,
    tag: str | None = None,
) -> dict:
    """Tokenize until ``target_tokens`` training tokens exist.  Cached."""
    import tiktoken

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or source
    meta_path = data_dir / f"{tag}_meta.json"
    train_path = data_dir / f"{tag}_train.bin"
    val_path = data_dir / f"{tag}_val.bin"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("n_train", 0) >= target_tokens and train_path.exists():
            return meta

    enc = tiktoken.get_encoding("gpt2")
    inv = packed.inv_order
    eot = enc.eot_token

    val_ids: list[np.ndarray] = []
    train_chunks: list[np.ndarray] = []
    n_train = 0
    n_docs = 0

    with open(train_path, "wb") as fh:
        for doc in iter_documents(source):
            ids = np.asarray(enc.encode_ordinary(doc) + [eot], dtype=np.int64)
            packed_ids = inv[ids].astype(np.uint32)
            n_docs += 1
            if n_docs <= N_VAL_DOCS:
                val_ids.append(packed_ids)
                continue
            train_chunks.append(packed_ids)
            n_train += packed_ids.size
            if len(train_chunks) >= 512:
                np.concatenate(train_chunks).tofile(fh)
                train_chunks = []
            if n_train >= target_tokens:
                break
        if train_chunks:
            np.concatenate(train_chunks).tofile(fh)

    val = np.concatenate(val_ids) if val_ids else np.zeros(0, dtype=np.uint32)
    val.tofile(val_path)

    meta = {
        "source": source,
        "tag": tag,
        "n_train": int(n_train),
        "n_val": int(val.size),
        "n_docs": int(n_docs),
        "n_val_docs": min(N_VAL_DOCS, n_docs),
        "vocab": int(packed.V),
        "note": "ids are in PackedVocab order (length-sorted), not raw GPT-2 order",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class TokenStream:
    """Memmapped uint32 token bin with deterministic batch sampling."""

    def __init__(self, path: Path, block_size: int, seed: int = 0) -> None:
        self.path = Path(path)
        self.data = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.data.size)

    def _windows(self, starts: np.ndarray):
        """One fancy-index gather rather than 2*B slices and two stacks.

        Batch fetch is on the critical path once the training loop stops
        syncing every step, so it is worth not doing it the obvious way.
        """
        span = np.arange(self.block_size + 1, dtype=np.int64)
        chunk = np.asarray(self.data[starts[:, None] + span[None, :]], dtype=np.int64)
        return chunk[:, :-1], chunk[:, 1:]

    def batch(self, batch_size: int):
        hi = len(self) - self.block_size - 1
        return self._windows(self.rng.integers(0, hi, size=batch_size))

    def sequential(self, batch_size: int, max_batches: int | None = None):
        """Non-overlapping windows, in order.  Used for evaluation."""
        step = self.block_size
        n = (len(self) - 1) // step
        if max_batches is not None:
            n = min(n, max_batches * batch_size)
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n), dtype=np.int64) * step
            if idx.size == 0:
                break
            yield self._windows(idx)


def resolve_paths(tag: str, data_dir: Path = DEFAULT_DATA_DIR) -> tuple[Path, Path]:
    data_dir = Path(data_dir)
    return data_dir / f"{tag}_train.bin", data_dir / f"{tag}_val.bin"


def env_device() -> str:
    import torch

    if os.environ.get("KRON_FORCE_CPU"):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
