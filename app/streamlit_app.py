import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from foodlens.classes import CLASSES
from foodlens.inference import load_torch, predict_pil

st.set_page_config(page_title="FoodLens", page_icon="🍜")
st.title("FoodLens")
st.caption("Upload a photo and see the model's three highest-scoring dishes.")
st.info(
    "This model only knows **ten dishes**: " + ", ".join(c.replace("_", " ") for c in CLASSES) + ". "
    "It always picks among these, so a photo of any other food can still get a high score. "
    "Scores are softmax outputs, not calibrated probabilities."
)


@st.cache_resource
def get_model(path):
    return load_torch(path)


ckpt = st.sidebar.text_input("Checkpoint", os.environ.get("FOODLENS_CKPT", "runs/pt_mnv3/best.pt"))
if not Path(ckpt).exists():
    st.error(f"Checkpoint not found: {ckpt}. Train a model first (see README).")
    st.stop()
model, preset = get_model(ckpt)

file = st.file_uploader("JPG or PNG", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file)
    st.image(img, caption="Your photo", use_container_width=True)
    for rank, (name, score) in enumerate(predict_pil(model, preset, img), 1):
        st.write(f"**{rank}. {name.replace('_', ' ')}** — {score:.1%}")
        st.progress(min(score, 1.0))
