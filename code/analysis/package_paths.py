"""External input locations and portable metadata paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = (
    Path(os.environ.get("ATHENA_DATA_ROOT", PACKAGE_ROOT / "data"))
    .expanduser()
    .resolve()
)
RESULT_ROOT = (
    Path(os.environ.get("ATHENA_OUTPUT_ROOT", PACKAGE_ROOT / "outputs"))
    .expanduser()
    .resolve()
)
DATA_ROOT = INPUT_ROOT / "analysis"
OUTPUT_ROOT = RESULT_ROOT / "analysis"


def missing_input_message(error: FileNotFoundError) -> str:
    """Explain how to supply an absent external input without a traceback."""
    missing = error.filename or str(error)
    return (
        f"Missing input: {missing}. Supply --data-root with the directory containing "
        "analysis/ and mechanics/, or set ATHENA_DATA_ROOT. "
        "See docs/INPUTS.md for the required files."
    )


def run_cli(action: Callable[[], Any], description: str | None = None) -> None:
    """Run a direct module entry with help and concise missing-input errors."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=(
            "Set ATHENA_DATA_ROOT and ATHENA_OUTPUT_ROOT before running this module. "
            "The run_all.py entry also accepts --data-root and --output-root."
        ),
    )
    parser.parse_args()
    try:
        action()
    except FileNotFoundError as error:
        parser.exit(2, f"error: {missing_input_message(error)}\n")


def relative_metadata(value: Any) -> Any:
    """Use logical data/outputs prefixes for paths under the selected roots."""
    if isinstance(value, dict):
        return {str(key): relative_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [relative_metadata(item) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for root, label in (
            (INPUT_ROOT, "data/"),
            (RESULT_ROOT, "outputs/"),
            (PACKAGE_ROOT, ""),
        ):
            prefix = root.as_posix() + "/"
            if normalized.casefold().startswith(prefix.casefold()):
                return label + normalized[len(prefix) :]
    return value
