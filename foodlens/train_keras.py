"""Keras CNN with the same split, schedule, flip policy and checkpoint rule as train_torch.

  python -m foodlens.train_keras --name keras_cnn
"""
import argparse
import csv
import time
from pathlib import Path

import keras
import numpy as np
from PIL import Image

from .data import preprocess_u8, read_manifest
from .keras_model import build_keras_cnn
from .metrics import compute_metrics
from .utils import env_info, save_json, seed_everything


def load_split(manifest, root, split):
    df = read_manifest(manifest, split)
    x = np.stack([preprocess_u8(Image.open(Path(root) / p), "cnn") for p in df.path])
    return x, df.class_id.to_numpy()


def to_float(x_u8):  # identical to data.preprocess_pil for the "cnn" preset
    return x_u8.astype(np.float32) / 255.0


def ce(logits, y):
    z = logits - logits.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--data-root", default="data/food-101")
    p.add_argument("--name", default="keras_cnn")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-flip", action="store_true")
    a = p.parse_args()

    seed_everything(a.seed)
    keras.utils.set_random_seed(a.seed)
    import tensorflow as tf

    devices = [d.name for d in tf.config.list_physical_devices()]
    cfg = {**vars(a), "devices": devices, "keras": keras.__version__, "tf": tf.__version__, "env": env_info()}
    print("devices:", devices)

    xtr, ytr = load_split(a.manifest, a.data_root, "train")
    xva, yva = load_split(a.manifest, a.data_root, "val")
    model = build_keras_cnn()
    model.compile(
        optimizer=keras.optimizers.AdamW(a.lr, weight_decay=a.weight_decay),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    run = Path("runs") / a.name
    run.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run / "config.json")
    log = open(run / "log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1", "val_top1", "seconds"])
    rng, best, t_start = np.random.default_rng(a.seed), -1.0, time.time()

    for ep in range(1, a.epochs + 1):
        t0, total = time.time(), 0.0
        order = rng.permutation(len(xtr))
        for i in range(0, len(order), a.batch_size):
            idx = order[i:i + a.batch_size]
            xb = xtr[idx].copy()
            if not a.no_flip:
                flip = rng.random(len(idx)) < 0.5
                xb[flip] = xb[flip][:, :, ::-1]
            total += float(model.train_on_batch(to_float(xb), ytr[idx])) * len(idx)
        logits = model.predict(to_float(xva), batch_size=64, verbose=0)
        m = compute_metrics(logits, yva)
        w.writerow([ep, total / len(xtr), ce(logits, yva), m["macro_f1"], m["top1"], round(time.time() - t0, 1)])
        log.flush()
        print(f"ep {ep:2d} train {total/len(xtr):.4f} val {ce(logits, yva):.4f} macroF1 {m['macro_f1']:.4f}")
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            model.save(run / "best.keras")
    save_json({**cfg, "best_val_macro_f1": best, "total_seconds": round(time.time() - t_start, 1)}, run / "config.json")
    print(f"best val macro-F1 {best:.4f} -> {run/'best.keras'}")


if __name__ == "__main__":
    main()
