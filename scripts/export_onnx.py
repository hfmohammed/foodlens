"""Optional: export a PyTorch checkpoint to ONNX, check parity on 50 validation inputs, then benchmark CPU latency.

  python scripts/export_onnx.py --checkpoint runs/pt_mnv3/best.pt --out results/model.onnx
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foodlens.data import load_image, read_manifest  # noqa: E402
from foodlens.inference import load_torch  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--out", default="results/model.onnx")
p.add_argument("--manifest", default="data/manifest.csv")
p.add_argument("--data-root", default="data/food-101")
p.add_argument("--atol", type=float, default=1e-4)
a = p.parse_args()

model, preset = load_torch(a.checkpoint)
df = read_manifest(a.manifest, "val").sample(50, random_state=0)
x = np.stack([load_image(Path(a.data_root) / q, preset).transpose(2, 0, 1) for q in df.path]).astype(np.float32)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(model, torch.from_numpy(x[:1]), a.out, input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
with torch.no_grad():
    ref = model(torch.from_numpy(x)).numpy()
got = sess.run(None, {"image": x})[0]
diff = np.abs(ref - got).max()
agree = (ref.argmax(1) == got.argmax(1)).mean()
print(f"max |logit diff| = {diff:.2e} (atol {a.atol:g}); top-1 agreement = {agree:.0%}")
assert diff <= a.atol and agree == 1.0, "parity failed - not benchmarking"


def bench(fn, n=100):
    for _ in range(10):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return np.percentile(t, 50), np.percentile(t, 95)


one = x[:1]
with torch.no_grad():
    print("PyTorch CPU  p50/p95 ms: %.2f / %.2f" % bench(lambda: model(torch.from_numpy(one))))
print("ONNX RT CPU  p50/p95 ms: %.2f / %.2f" % bench(lambda: sess.run(None, {"image": one})))
