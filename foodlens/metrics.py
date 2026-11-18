import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from .classes import NUM_CLASSES


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """probs: (N, 10) scores (softmax or logits - only ranking is used). labels: (N,)."""
    labels = np.asarray(labels)
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    ids = list(range(NUM_CLASSES))
    return dict(
        top1=float((pred == labels).mean()),
        top3=float((top3 == labels[:, None]).any(1).mean()),
        macro_f1=float(f1_score(labels, pred, average="macro", labels=ids, zero_division=0)),
        per_class_recall=recall_score(labels, pred, average=None, labels=ids, zero_division=0).tolist(),
        confusion_matrix=confusion_matrix(labels, pred, labels=ids).tolist(),
    )
