# FoodLens

Photo classifier for ten dishes from Food-101. Upload an image, get the three highest-scoring classes with model scores.
Trains a small CNN end to end in PyTorch, reproduces it in Keras, and compares both against a transfer-learned MobileNetV3-Small, all on the same split.

**Classes:** hamburger, hot_dog, pizza, tacos, breakfast_burrito, ramen, pho, sushi, dumplings, gyoza (see [configs/classes.txt](configs/classes.txt); line number = class ID). Dumplings vs. gyoza is the pair to inspect in the confusion matrix.

> Results below are placeholders (`TBD`) until the runs are done. Fill them in with measured numbers only.

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (~5 GB; not included in this repo - download it from the original source and extract to `data/food-101/`).

> Bossard, L., Guillaumin, M., Van Gool, L. *Food-101 – Mining Discriminative Components with Random Forests.* European Conference on Computer Vision (ECCV), 2014.

Download and extract (see the dataset page above for the current archive link):

```bash
curl -L -o data/food-101.tar.gz https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf data/food-101.tar.gz -C data/    # creates data/food-101/{images,meta}
```

Split: the official 750 train images per class become 600 train / 150 val using seed 42; the official 250 test images per class are the test set. Totals: 6,000 / 1,500 / 2,500.

```bash
python scripts/make_split.py --food101-dir data/food-101 --seed 42   # writes data/manifest.csv (path, class_id, split)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-extra.txt   # Keras comparison + optional ONNX
pytest
```

## Preprocessing (identical in training, evaluation, serving)

[foodlens/data.py](foodlens/data.py): convert to RGB, PIL bilinear resize, scale to [0, 1]. Random horizontal flip is training-only.

| Model | Input |
|---|---|
| CNN (PyTorch, Keras) | squash-resize to 128 x 128, [0, 1] |
| MobileNetV3-Small | resize 256, center-crop 224, [0, 1], ImageNet mean/std (what the pretrained weights require) |

## Train

```bash
# 1. sanity check: overfit one batch of 32 (loss should approach 0)
python -m foodlens.train_torch --arch cnn --overfit-batch

# 2. PyTorch CNN: AdamW, lr 1e-3, batch 32, 15 epochs, best val macro-F1 checkpoint
python -m foodlens.train_torch --arch cnn --name pt_cnn

# 3. Keras twin: same split, architecture, scaling, schedule, checkpoint rule
python -m foodlens.train_keras --name keras_cnn

# 4. MobileNetV3-Small: frozen backbone, train head
python -m foodlens.train_torch --arch mobilenet --name pt_mnv3
# 4b. optional: unfreeze last feature block, lower LR
python -m foodlens.train_torch --arch mobilenet --unfreeze-last --lr 1e-4 --epochs 10 \
    --init-from runs/pt_mnv3/best.pt --name pt_mnv3_ft
```

Each run writes `runs/<name>/{config.json, log.csv, best.*}` (seed, hyperparameters, device used, library versions; per-epoch train loss, val loss, val macro-F1). Checkpoints are git-ignored.

## Evaluate

Pick checkpoints and augmentation using `--split val`. Run `--split test` once, after those choices are locked.

```bash
python -m foodlens.evaluate --framework torch --checkpoint runs/pt_cnn/best.pt   --split test --out results/pt_cnn
python -m foodlens.evaluate --framework keras --checkpoint runs/keras_cnn/best.keras --split test --out results/keras_cnn
python -m foodlens.evaluate --framework torch --checkpoint runs/pt_mnv3/best.pt  --split test --out results/pt_mnv3
python scripts/make_results_table.py
```

Outputs per model: `metrics.json` (top-1, top-3, macro-F1, per-class recall, confusion matrix), `confusion_matrix.png`, `mistakes.csv` (10 most confident errors with true label, predicted label, score).

## Results (official test set, 2,500 images)

| Model | Device | Top-1 | Top-3 | Macro-F1 |
|---|---|---|---|---|
| PyTorch CNN | TBD | TBD | TBD | TBD |
| Keras CNN | TBD | TBD | TBD | TBD |
| MobileNetV3-Small (frozen head) | TBD | TBD | TBD | TBD |

Per-class recall, confusion matrix, and example mistakes: TBD (`results/*/`).

### PyTorch vs. Keras: differences that affect interpretation

- **Initialization:** PyTorch conv/linear default is Kaiming-uniform; Keras default is Glorot-uniform. Not matched.
- **Device:** TBD (record MPS/CPU for PyTorch and what TensorFlow used). Wall-clock times are only comparable if both ran on the same device.
- **Data order / flips:** both seeded, but the RNG streams differ, so batches and flips are not identical.
- **AdamW:** same lr and weight decay (0.01); implementations are near-identical but not bit-identical.

## Demo

```bash
FOODLENS_CKPT=runs/pt_mnv3/best.pt streamlit run app/streamlit_app.py
```

The page takes a JPG/PNG upload, previews it, shows three ranked predictions, and warns that only ten dishes are supported and unsupported foods may still get high scores.

Screenshot: TBD (`docs/demo.png`).

**Model chosen for the demo:** TBD, with the reason based on validation/test results.

## Optional: ONNX parity

```bash
python scripts/export_onnx.py --checkpoint runs/pt_mnv3/best.pt
```

Compares PyTorch and ONNX Runtime logits on 50 validation inputs (tolerance `--atol`, default 1e-4) and top-1 agreement; benchmarks CPU latency only if parity passes. Results: TBD.

## Tests

`pytest` covers class ID mapping, preprocessing (shape, range, RGB conversion, normalization), model output shapes, and split reproducibility/sizes.
