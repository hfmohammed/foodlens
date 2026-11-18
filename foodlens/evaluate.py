"""Evaluate a checkpoint. Run on --split test ONLY after checkpoint/augmentation choices are locked.

  python -m foodlens.evaluate --framework torch --checkpoint runs/pt_cnn/best.pt --split val --out results/pt_cnn_val
  python -m foodlens.evaluate --framework keras --checkpoint runs/keras_cnn/best.keras --split test --out results/keras_cnn
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .classes import CLASSES
from .data import ManifestDataset, preprocess_u8, read_manifest
from .metrics import compute_metrics
from .utils import pick_device, save_json


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def torch_probs(ckpt, manifest, root, split, device):
    from .inference import load_torch

    model, preset = load_torch(ckpt, device)
    ds = ManifestDataset(manifest, root, split, preset)
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 64):
            xs = torch.stack([ds[j][0] for j in range(i, min(i + 64, len(ds)))]).to(device)
            out.append(model(xs).cpu().numpy())
    return softmax(np.concatenate(out))


def keras_probs(ckpt, manifest, root, split):
    import keras

    df = read_manifest(manifest, split)
    x = np.stack([preprocess_u8(Image.open(Path(root) / p), "cnn") for p in df.path]).astype(np.float32) / 255.0
    return softmax(keras.saving.load_model(ckpt).predict(x, batch_size=64, verbose=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["torch", "keras"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--data-root", default="data/food-101")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    df = read_manifest(a.manifest, a.split)
    probs = (torch_probs(a.checkpoint, a.manifest, a.data_root, a.split, pick_device(a.device))
             if a.framework == "torch" else keras_probs(a.checkpoint, a.manifest, a.data_root, a.split))
    y = df.class_id.to_numpy()
    m = compute_metrics(probs, y)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json({**m, "split": a.split, "checkpoint": a.checkpoint, "classes": CLASSES}, out / "metrics.json")

    pred = probs.argmax(1)
    wrong = df.assign(true=[CLASSES[i] for i in y], pred=[CLASSES[i] for i in pred], score=probs.max(1))[pred != y]
    wrong.sort_values("score", ascending=False).head(10)[["path", "true", "pred", "score"]].to_csv(out / "mistakes.csv", index=False)

    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=60, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)

    print(f"{a.split}: top1 {m['top1']:.4f} top3 {m['top3']:.4f} macroF1 {m['macro_f1']:.4f}")
    for c, r in zip(CLASSES, m["per_class_recall"]):
        print(f"  {c:18s} recall {r:.3f}")


if __name__ == "__main__":
    main()
