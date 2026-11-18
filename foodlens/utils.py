import json
import platform
import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def pick_device(name: str = "auto") -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def env_info() -> dict:
    info = dict(python=platform.python_version(), platform=platform.platform(), machine=platform.machine())
    try:
        import torch

        info["torch"] = str(torch.__version__)
    except ImportError:
        pass
    return info


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def load_checkpoint(path):
    """Load our own checkpoints with weights_only=True, allowing only the TorchVersion type that
    checkpoints written before env_info() stored a plain string contain."""
    import torch
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)
