"""Class list: configs/classes.txt is the single source of truth (line number = class ID)."""
from pathlib import Path

CLASSES_FILE = Path(__file__).resolve().parent.parent / "configs" / "classes.txt"


def load_classes(path: Path = CLASSES_FILE) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


CLASSES = load_classes()
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = dict(enumerate(CLASSES))
NUM_CLASSES = len(CLASSES)
