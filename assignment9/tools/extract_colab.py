"""Extract an independent Colab run out of its executed notebook, mechanically.

    python tools/extract_colab.py <path-to-colab-executed-loss_harness.ipynb>

Writes results/colab_run_log.txt (that run's full stdout) and results/colab_summary.json
(the numbers tools/compare_colab.py diffs against results/results.json).

Two things this refuses to do, because a cross-run comparison that skips them is theatre:

* It refuses a notebook whose code cells are not byte-identical to the committed
  loss_harness.ipynb. Comparing numbers from a different program proves nothing.
* It refuses a notebook that did not run cleanly end to end -- execution_count must be
  1..N with no gaps, and no cell may carry an error output.

Every number is pulled by an anchored regex against the harness's own print format, and a
pattern that fails to match is a hard error rather than a silently missing key.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMITTED = os.path.join(HERE, "loss_harness.ipynb")
LOG_OUT = os.path.join(HERE, "results", "colab_run_log.txt")
JSON_OUT = os.path.join(HERE, "results", "colab_summary.json")

NUM = r"([-+]?[\d,]+\.?\d*(?:e[-+]?\d+)?)"


def num(s):
    s = s.replace(",", "")
    f = float(s)
    return int(f) if f.is_integer() and "." not in s and "e" not in s else f


# key -> regex with one capture group, matched against the whole log.
PATTERNS = {
    "env_python":              r"python\s+:\s*(\S+)",
    "env_torch":               r"torch\s+:\s*(\S+)",
    "env_gpu":                 r"gpu\s+:\s*(.+?)\s*\(",
    "env_gpu_gib":             r"gpu\s+:.+?\(" + NUM + r" GiB\)",

    "item1_hidden_numel":      r"hidden holds " + NUM + r" numbers",
    "item1_logits_numel":      r"logits holds " + NUM + r" —",
    "item1_blowup":            r"a factor of V/D = " + NUM + r"x",

    "item3_tokens_counted":    r"B\*\(T-1\) = " + NUM + r" positions exist",
    "item3_tokens_masked":     r"positions exist; only " + NUM + r" of them are real",
    "item3_pad_fraction_pct":  NUM + r"% of the naive denominator is padding",

    "item3b_final_counted":    r"final counted loss.*?:\s*" + NUM,
    "item3b_final_masked":     r"final masked loss.*?:\s*" + NUM,
    "item3b_gap":              r"gap\s+:\s*" + NUM + r" nats",

    "item4_n_before":          r"before masking the boundary\s+" + NUM + r"\s",
    "item4_loss_before":       r"before masking the boundary\s+[\d,]+\s+" + NUM,
    "item4_n_after":           r"after masking the boundary\s+" + NUM + r"\s",
    "item4_loss_after":        r"after masking the boundary\s+[\d,]+\s+" + NUM,
    "item4_boundary_term":     r"last token of A\) = " + NUM + r" nats",
    "item4_mean_boundary":     r"mean loss at the boundary site\s+:\s*" + NUM,
    "item4_mean_interior":     r"mean loss at every other site\s+:\s*" + NUM,
    "item4_mean_ratio":        r"ratio\s+:\s*" + NUM + r"x",
    "item4_boundary_worse_pct": r"own sequence average in\s*" + NUM + r"% of pairs",

    "item5_init_loss":         r"loss at initialisation\s+:\s*" + NUM + r" nats",
    "item5_init_ppl":          r"perplexity\s+:\s*" + NUM,
    "item5_ppl_over_vocab":    r"perplexity / V\s+:\s*" + NUM,
    "item5_uniform_control":   r"uniform-logits control\s+:\s*" + NUM + r" nats",

    "item6_body_params":       r"transformer body \(4 blocks \+ norm\)\s+" + NUM,
    "item6_embed_params":      r"input embedding\s+\[V, D\]\s+" + NUM,
    "item6_untied_total":      r"^UNTIED total\s+" + NUM,
    "item6_tied_total":        r"^TIED total\s+" + NUM,   # anchored: "UNTIED total" contains it
    "item6_saved":             r"saved by tying\s+" + NUM,
    "item6_saved_pct":         r"saved by tying\s+[\d,]+\s+\(" + NUM + r"% of the model\)",

    "item7_loss_absdiff":      r"\|difference\|\s+:\s*" + NUM,
    "item7_peak_naive_mib":    r"ordinary cross-entropy\s+" + NUM + r" MiB",
    "item7_peak_chunked_mib":  r"chunked cross-entropy\s+" + NUM + r" MiB",
    "item7_ratio":             r"^ratio\s+" + NUM + r"x",

    "part2_head1_loss":        r"head 1  \(predicting t\+1\)\s+" + NUM,
    "part2_head2_loss":        r"head 2  \(predicting t\+2\)\s+" + NUM,
    "part2_sum":               r"sum  \(the quantity being optimised\)\s+" + NUM,
    "part2_gap":               r"gap  \(head 2 - head 1\)\s+" + NUM,
    "part2_baseline_head1_loss": r"single-head baseline, head 1\s+" + NUM,
    "part2_head1_delta_vs_baseline": r"effect of the extra head on head 1\s+" + NUM,

    "part3_bug_final_loss":    r"\[P3\].*?loss " + NUM,
    "part3_bug_final_ppl":     r"\[P3\].*?\(ppl " + NUM + r"\)",
    "part3_correct_final_loss": r"the correct shift sits at " + NUM,
    "part3_copy_rate_pct":     r"copy rate over the whole sequence: " + NUM + r"%",
}

SWEEP_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d,]+\.\d)\s+([\d.]+)x\s+([\d.]+)\s*$", re.M)


def load_notebook(path):
    nb = json.loads(io.open(path, encoding="utf-8").read())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    counts = [c.get("execution_count") for c in code]
    if counts != list(range(1, len(code) + 1)):
        raise SystemExit(f"{path}: cells did not run cleanly in order (execution_count={counts})")
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise SystemExit(f"{path}: {len(errors)} cell(s) raised; this is not a clean run")
    return nb, code


def assert_same_code(theirs):
    mine = json.loads(io.open(COMMITTED, encoding="utf-8").read())
    a = ["".join(c["source"]) for c in theirs]
    b = ["".join(c["source"]) for c in mine["cells"] if c["cell_type"] == "code"]
    if a != b:
        first = next((n for n, (x, y) in enumerate(zip(a, b), 1) if x != y), "length")
        raise SystemExit(f"code cells differ from the committed notebook (first at {first}); "
                         "comparing numbers from a different program proves nothing")
    return len(a)


def stdout_of(nb):
    return "".join("".join(o.get("text", []))
                   for c in nb["cells"] for o in c.get("outputs", [])
                   if o.get("output_type") == "stream")


def parse(log):
    out, missing = {}, []
    for key, pat in PATTERNS.items():
        m = re.search(pat, log, re.M | re.S)
        if not m:
            missing.append(key)
            continue
        raw = m.group(1)
        out[key] = raw if key in ("env_python", "env_torch", "env_gpu") else num(raw)
    if missing:
        raise SystemExit(f"these patterns matched nothing -- the print format changed? {missing}")

    block = log[log.index("Peak memory against chunk size"):log.index("Every row is the same")]
    sweep = [{"chunk": int(c), "chunks": int(n), "peak_mib": num(p),
              "ratio": float(r), "loss": float(l)}
             for c, n, p, r, l in SWEEP_ROW.findall(block)]
    if len(sweep) != 6:
        raise SystemExit(f"expected 6 sweep rows, parsed {len(sweep)}")
    out["item7_sweep"] = sweep
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = sys.argv[1]
    nb, code = load_notebook(src)
    n_code = assert_same_code(code)
    log = stdout_of(nb)
    io.open(LOG_OUT, "w", encoding="utf-8", newline="\n").write(log)

    data = parse(log)
    data = {
        "_source": (f"Extracted by tools/extract_colab.py from {os.path.basename(src)}, an "
                    "independent Google Colab run of loss_harness.ipynb. Its code cells were "
                    "verified byte-identical to the committed notebook before anything was "
                    "read out of it, and it ran cleanly: execution_count 1.."
                    f"{n_code} with no gaps and no error outputs."),
        "_full_stdout": "results/colab_run_log.txt",
        **data,
    }
    io.open(JSON_OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(data, indent=2) + "\n")
    print(f"verified {n_code} code cells identical to the committed notebook")
    print(f"wrote {LOG_OUT} ({len(log):,} chars)")
    print(f"wrote {JSON_OUT} ({len(PATTERNS)} scalars + a {len(data['item7_sweep'])}-row sweep)")
    print(f"  {data['env_gpu']}, python {data['env_python']}, torch {data['env_torch']}")


if __name__ == "__main__":
    main()
