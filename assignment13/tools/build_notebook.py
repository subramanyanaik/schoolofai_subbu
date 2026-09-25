"""Turn reversible_llm.py (jupytext 'percent' format) into reversible_llm.ipynb.

The .py file is the single source of truth. Cells are delimited by `# %%` for code and
`# %% [markdown]` for prose; markdown cells have their leading `# ` stripped.

    python tools/build_notebook.py             # rebuild the notebook (no outputs)
    python tools/build_notebook.py --execute   # rebuild, then run it and embed the outputs
    python tools/build_notebook.py --check     # exit 1 if the committed notebook's code
                                               # differs from reversible_llm.py (outputs ignored)
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "reversible_llm.py")
DST = os.path.join(HERE, "reversible_llm.ipynb")

BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/subramanyanaik/schoolofai_subbu/blob/main/"
    "assignment13/reversible_llm.ipynb)\n"
    "\n"
    "Generated from [`reversible_llm.py`](reversible_llm.py) by `tools/build_notebook.py` - "
    "edit the `.py`, not this file. **Use a GPU runtime**: Runtime -> Change runtime type -> "
    "T4 GPU (or better).\n"
)


def split_cells(text):
    cells, kind, buf = [], "code", []
    for line in text.split("\n"):
        stripped = line.rstrip()
        if stripped == "# %%" or stripped.startswith("# %% "):
            if buf:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
            continue
        buf.append(line)
    if buf:
        cells.append((kind, buf))
    return cells


def clean_markdown(lines):
    out = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return out


def trim(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def to_source(lines):
    """nbformat stores a cell as a list of lines, each ending in a newline but the last."""
    if not lines:
        return []
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def build_dict():
    text = io.open(SRC, encoding="utf-8").read()
    cells = [{"cell_type": "markdown", "metadata": {},
              "source": to_source(BADGE.rstrip("\n").split("\n"))}]
    for kind, lines in split_cells(text):
        lines = trim(list(lines))
        if not lines:
            continue
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": to_source(trim(clean_markdown(lines)))})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": to_source(lines)})
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def build():
    nb = build_dict()
    io.open(DST, "w", encoding="utf-8", newline="\n").write(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"wrote {DST}: {len(nb['cells'])} cells "
          f"({n_md} markdown, {len(nb['cells']) - n_md} code)")
    return DST


def sources(nb):
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def check():
    """Compare the committed notebook's cell sources against a fresh build.

    Outputs are deliberately ignored: the committed notebook may carry a real run's outputs,
    but its *code* has to be exactly what reversible_llm.py says, or the two have drifted.
    """
    committed = sources(json.loads(io.open(DST, encoding="utf-8").read()))
    expected = sources(build_dict())
    if committed != expected:
        for n in range(max(len(committed), len(expected))):
            a = committed[n] if n < len(committed) else None
            b = expected[n] if n < len(expected) else None
            if a != b:
                print(f"first difference at cell {n}:")
                print("  committed:", repr(a[1][:120] if a else None))
                print("  from .py :", repr(b[1][:120] if b else None))
                break
        print("reversible_llm.ipynb is out of sync with reversible_llm.py")
        print("Run: python tools/build_notebook.py")
        return 1
    print(f"reversible_llm.ipynb is in sync with reversible_llm.py ({len(expected)} cells)")
    return 0


def execute(path):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=14400, kernel_name="python3",
                   resources={"metadata": {"path": HERE}}).execute()
    nbformat.write(nb, path)
    print(f"executed and saved outputs into {path}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        raise SystemExit(check())
    out = build()
    if "--execute" in sys.argv:
        execute(out)
