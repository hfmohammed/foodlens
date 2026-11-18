"""Collect results/*/metrics.json (test split) into a markdown table."""
import json
import sys
from pathlib import Path

rows = []
for f in sorted(Path("results").glob("*/metrics.json")):
    m = json.loads(f.read_text())
    if m["split"] != "test":
        continue
    rows.append(f"| {f.parent.name} | {m['top1']:.3f} | {m['top3']:.3f} | {m['macro_f1']:.3f} |")
print("| Model | Top-1 | Top-3 | Macro-F1 |\n|---|---|---|---|\n" + "\n".join(rows))
sys.exit(0 if rows else 1)
