"""Build the out-of-domain multi-script corpus E7 measures on.

The reference run read ``../assignment2/data/*.faithful.txt``, which does not
exist in this repository.  This rebuilds an equivalent corpus from Wikipedia
so E7 is reproducible on any machine with network access.

    python experiments/build_ood_corpus.py

Trap 9: every read and write here passes encoding="utf-8" explicitly.  On
Windows the default is cp1252 and Devanagari/Telugu raise UnicodeEncodeError.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, rule, table
from kronecker_v2.data import force_utf8_stdout

force_utf8_stdout()

OOD_DIR = ROOT / "data" / "ood"
LANGS = {
    "en": ("20231101.en", "English"),
    "es": ("20231101.es", "Spanish"),
    "hi": ("20231101.hi", "Hindi (Devanagari)"),
    "te": ("20231101.te", "Telugu"),
}


def build(lang: str, config: str, target_chars: int) -> Path:
    import datasets

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    out = OOD_DIR / f"{lang}.faithful.txt"
    if out.exists() and out.stat().st_size >= target_chars // 2:
        return out
    ds = datasets.load_dataset(
        "wikimedia/wikipedia", config, split="train", streaming=True
    )
    chunks, total = [], 0
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        chunks.append(text)
        total += len(text)
        if total >= target_chars:
            break
    out.write_text("\n\n".join(chunks), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=400_000,
                    help="characters per language")
    args = ap.parse_args()

    rule("building the out-of-domain multi-script corpus (Wikipedia)")
    rows, meta = [], {}
    for lang, (config, label) in LANGS.items():
        path = build(lang, config, args.chars)
        text = path.read_text(encoding="utf-8")
        n_bytes = len(text.encode("utf-8"))
        meta[lang] = {
            "label": label, "config": config, "path": str(path.relative_to(ROOT)),
            "chars": len(text), "bytes": n_bytes,
            "bytes_per_char": n_bytes / max(len(text), 1),
        }
        rows.append([lang, label, f"{len(text):,}", f"{n_bytes:,}",
                     f"{meta[lang]['bytes_per_char']:.2f}"])
    table(rows, ["lang", "script", "chars", "utf-8 bytes", "bytes/char"])
    (OOD_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n-> {OOD_DIR.relative_to(ROOT)}")
    print("bytes/char is why this experiment matters: Devanagari and Telugu cost")
    print("~3 UTF-8 bytes per character, and GPT-2 BPE has almost no merges for them.")


if __name__ == "__main__":
    main()
