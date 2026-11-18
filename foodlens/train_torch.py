"""Train the PyTorch CNN or MobileNetV3-Small head.

  python -m foodlens.train_torch --arch cnn --overfit-batch          # sanity check
  python -m foodlens.train_torch --arch cnn --name pt_cnn            # full run
  python -m foodlens.train_torch --arch mobilenet --name pt_mnv3     # frozen backbone, train head
  python -m foodlens.train_torch --arch mobilenet --unfreeze-last --lr 1e-4 --name pt_mnv3_ft
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import ManifestDataset
from .metrics import compute_metrics
from .models import build_model, set_train_mode
from .utils import env_info, load_checkpoint, pick_device, save_json, seed_everything


def predict_logits(model, loader, device):
    model.eval()
    out, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            out.append(model(x.to(device)).float().cpu())
            ys.append(y)
    return torch.cat(out), torch.cat(ys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["cnn", "mobilenet"], default="cnn")
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--data-root", default="data/food-101")
    p.add_argument("--name", default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-flip", action="store_true", help="disable the training-time horizontal flip")
    p.add_argument("--unfreeze-last", action="store_true", help="mobilenet: also train the last feature block")
    p.add_argument("--init-from", default=None, help="checkpoint to start from (e.g. the frozen-head run when fine-tuning)")
    p.add_argument("--overfit-batch", action="store_true", help="sanity check: overfit one batch of 32, then exit")
    a = p.parse_args()

    seed_everything(a.seed)
    device = pick_device(a.device)
    preset = a.arch
    run = Path("runs") / (a.name or f"pt_{a.arch}")
    cfg = {**vars(a), "device_used": device, "env": env_info()}
    print("device:", device)

    train_ds = ManifestDataset(a.manifest, a.data_root, "train", preset, train=not a.no_flip and not a.overfit_batch)
    val_ds = ManifestDataset(a.manifest, a.data_root, "val", preset)
    g = torch.Generator().manual_seed(a.seed)
    model = build_model(a.arch, unfreeze_last=a.unfreeze_last).to(device)
    if a.init_from:
        model.load_state_dict(load_checkpoint(a.init_from)["state_dict"])
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    loss_fn = nn.CrossEntropyLoss()  # takes raw logits

    if a.overfit_batch:
        # manifest is ordered by class, so draw a random batch (seeded) to cover many classes
        idx = torch.randperm(len(train_ds), generator=g)[:32].tolist()
        x, y = next(iter(DataLoader(Subset(train_ds, idx), batch_size=32)))
        print("classes in batch:", sorted(set(y.tolist())))
        x, y = x.to(device), y.to(device)
        for step in range(1, 501):
            set_train_mode(model, a.arch, a.unfreeze_last)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            gn = sum(q.grad.norm() ** 2 for q in params if q.grad is not None) ** 0.5
            opt.step()
            if step % 50 == 0 or step == 1:
                print(f"step {step:3d} loss {loss.item():.4f} grad-norm {gn:.3f}")
        print("PASS" if loss.item() < 0.05 else "FAIL: loss did not approach 0 - check labels/loss/gradients")
        return

    tl = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers, generator=g, persistent_workers=a.workers > 0)
    vl = DataLoader(val_ds, 64, num_workers=a.workers)
    run.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run / "config.json")
    log = open(run / "log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1", "val_top1", "seconds"])
    best, t_start = -1.0, time.time()

    for ep in range(1, a.epochs + 1):
        t0, total, n = time.time(), 0.0, 0
        set_train_mode(model, a.arch, a.unfreeze_last)
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        logits, ys = predict_logits(model, vl, device)
        vloss = loss_fn(logits, ys).item()
        m = compute_metrics(logits.numpy(), ys.numpy())
        w.writerow([ep, total / n, vloss, m["macro_f1"], m["top1"], round(time.time() - t0, 1)])
        log.flush()
        print(f"ep {ep:2d} train {total/n:.4f} val {vloss:.4f} macroF1 {m['macro_f1']:.4f}")
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(dict(arch=a.arch, preset=preset, state_dict=model.state_dict(), epoch=ep, val_macro_f1=best, config=cfg), run / "best.pt")
    save_json({**cfg, "best_val_macro_f1": best, "total_seconds": round(time.time() - t_start, 1)}, run / "config.json")
    print(f"best val macro-F1 {best:.4f} -> {run/'best.pt'}")


if __name__ == "__main__":
    main()
