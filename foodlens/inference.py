"""Model loading + top-k prediction, used by evaluate.py and the Streamlit app."""
import numpy as np
import torch

from .classes import CLASSES
from .data import preprocess_pil
from .models import build_model


def load_torch(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["preset"]


def predict_pil(model, preset: str, img, k: int = 3, device: str = "cpu"):
    x = torch.from_numpy(preprocess_pil(img, preset).transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].cpu().numpy()
    top = np.argsort(-probs)[:k]
    return [(CLASSES[i], float(probs[i])) for i in top]
