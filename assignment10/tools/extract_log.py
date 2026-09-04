"""Write results/run_log.txt from the executed notebook's own stdout.

Taking the log out of the notebook rather than out of a separate script run means the log,
the figures and results/results.json all come from one execution and cannot disagree.
"""
import io
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(HERE, "truth_harness.ipynb")
LOG = os.path.join(HERE, "results", "run_log.txt")

nb = json.loads(io.open(NB, encoding="utf-8").read())
executed = sum(1 for c in nb["cells"] if c.get("outputs"))
if not executed:
    raise SystemExit("truth_harness.ipynb has no outputs -- run "
                     "`python tools/build_notebook.py --execute` first")

parts = []
for cell in nb["cells"]:
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append("".join(out.get("text", [])))
        elif out.get("output_type") == "error":
            parts.append("\n".join(out.get("traceback", [])) + "\n")

io.open(LOG, "w", encoding="utf-8", newline="\n").write("".join(parts))
print(f"wrote {LOG} from {executed} executed cells ({len(''.join(parts)):,} chars)")
