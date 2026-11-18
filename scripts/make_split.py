"""Generate data/manifest.csv. Usage: python scripts/make_split.py --food101-dir /path/to/food-101"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foodlens.split import make_manifest  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--food101-dir", required=True, type=Path)
p.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

df = make_manifest(a.food101_dir, a.seed)
a.out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(df.split.value_counts().to_string())
print(f"wrote {a.out} ({len(df)} rows, seed={a.seed})")
