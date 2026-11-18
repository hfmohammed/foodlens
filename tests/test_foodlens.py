import numpy as np
import torch
from PIL import Image

from foodlens.classes import CLASS_TO_ID, CLASSES, ID_TO_CLASS, NUM_CLASSES
from foodlens.data import preprocess_pil
from foodlens.models import FoodCNN, build_mobilenet
from foodlens.split import make_manifest


def test_class_mapping():
    assert NUM_CLASSES == 10 and len(set(CLASSES)) == 10
    assert CLASSES[0] == "hamburger" and CLASSES[-1] == "gyoza"
    assert all(ID_TO_CLASS[CLASS_TO_ID[c]] == c for c in CLASSES)


def test_cnn_preprocessing_shape_range_rgb():
    for mode in ("L", "RGBA", "RGB"):
        x = preprocess_pil(Image.new(mode, (300, 200)), "cnn")
        assert x.shape == (128, 128, 3) and x.dtype == np.float32
        assert 0.0 <= x.min() and x.max() <= 1.0
    white = preprocess_pil(Image.new("RGB", (50, 50), (255, 255, 255)), "cnn")
    assert np.allclose(white, 1.0)


def test_mobilenet_preprocessing_shape_and_normalization():
    x = preprocess_pil(Image.new("RGB", (400, 300), (0, 0, 0)), "mobilenet")
    assert x.shape == (224, 224, 3)
    assert np.allclose(x[0, 0], -np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225]), atol=1e-5)


def test_cnn_output_shape():
    assert FoodCNN()(torch.zeros(2, 3, 128, 128)).shape == (2, 10)


def test_mobilenet_output_shape_and_freezing():
    m = build_mobilenet(pretrained=False)
    assert m(torch.zeros(1, 3, 224, 224)).shape == (1, 10)
    assert not any(p.requires_grad for p in m.features.parameters())


def test_split_is_reproducible_and_sized(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    tr = "\n".join(f"{c}/{i}" for c in CLASSES for i in range(750))
    te = "\n".join(f"{c}/t{i}" for c in CLASSES for i in range(250))
    (meta / "train.txt").write_text(tr)
    (meta / "test.txt").write_text(te)
    a, b = make_manifest(tmp_path, 42), make_manifest(tmp_path, 42)
    assert a.equals(b)
    assert a.split.value_counts().to_dict() == {"train": 6000, "val": 1500, "test": 2500}
    assert not set(a[a.split == "train"].path) & set(a[a.split == "val"].path)
    assert not make_manifest(tmp_path, 7).equals(a)
