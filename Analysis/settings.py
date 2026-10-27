"""Input and output directories for electrode analysis."""

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "results"


def relative_metadata(value: Any) -> Any:
    """Express file locations relative to the analysis directories."""
    if isinstance(value, dict):
        return {str(key): relative_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [relative_metadata(item) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for root, label in (
            (DATA_ROOT, "data/"),
            (OUTPUT_ROOT, "results/"),
            (ROOT, ""),
        ):
            prefix = root.as_posix() + "/"
            if normalized.casefold().startswith(prefix.casefold()):
                return label + normalized[len(prefix) :]
    return value
