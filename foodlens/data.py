"""Preprocessing shared by training, evaluation and serving (PyTorch, Keras, Streamlit, ONNX).

Every path goes through `preprocess_pil`, so interpolation (PIL bilinear) is identical everywhere.
Output is float32 HWC.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# cnn: squash to 128x128, pixels in [0, 1].
# mobilenet: what torchvision's MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms() does
#            (bilinear resize 256, center-crop 224, [0,1], ImageNet mean/std).
PRESETS = {
    "cnn": dict(size=128, resize=None, mean=None, std=None),
    "mobilenet": dict(size=224, resize=256, mean=IMAGENET_MEAN, std=IMAGENET_STD),
}


def preprocess_u8(img: Image.Image, preset: str = "cnn") -> np.ndarray:
    """RGB + resize (+ center crop) -> uint8 HWC."""
    cfg = PRESETS[preset]
    img = img.convert("RGB")
    if cfg["resize"] is None:
        img = img.resize((cfg["size"], cfg["size"]), Image.BILINEAR)
    else:
        w, h = img.size
        s = cfg["resize"] / min(w, h)
        img = img.resize((max(cfg["resize"], round(w * s)), max(cfg["resize"], round(h * s))), Image.BILINEAR)
        w, h = img.size
        L, T = (w - cfg["size"]) // 2, (h - cfg["size"]) // 2
        img = img.crop((L, T, L + cfg["size"], T + cfg["size"]))
    return np.asarray(img, dtype=np.uint8)


def preprocess_pil(img: Image.Image, preset: str = "cnn") -> np.ndarray:
    x = preprocess_u8(img, preset).astype(np.float32) / 255.0
    cfg = PRESETS[preset]
    if cfg["mean"] is not None:
        x = (x - np.array(cfg["mean"], np.float32)) / np.array(cfg["std"], np.float32)
    return x


def load_image(path, preset: str = "cnn") -> np.ndarray:
    with Image.open(path) as im:
        return preprocess_pil(im, preset)


def read_manifest(manifest: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    return df[df.split == split].reset_index(drop=True)


class ManifestDataset:
    """torch Dataset (duck-typed; yields CHW float tensors). Random horizontal flip only if train=True."""

    def __init__(self, manifest: str, root: str, split: str, preset: str = "cnn", train: bool = False):
        self.df, self.root, self.preset, self.train = read_manifest(manifest, split), Path(root), preset, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import torch

        row = self.df.iloc[i]
        x = load_image(self.root / row.path, self.preset)
        if self.train and random.random() < 0.5:
            x = x[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), int(row.class_id)
