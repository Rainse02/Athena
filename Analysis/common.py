"""Image loading, binary masks, array division and JSON output."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, TypeVar

import cv2
import numpy as np

from settings import relative_metadata


T = TypeVar("T")


def natural_sort(values: Iterable[T]) -> list[T]:
    """Sort strings or path-like values using human numeric ordering.

    For example, ``slice2.tif`` precedes ``slice10.tif``.  The original value
    objects are returned, so callers can preserve their chosen path type.
    """

    def key(value: T) -> list[int | str]:
        text = str(value)
        parts = re.split(r"(\d+)", text)
        return [int(part) if part.isdigit() else part.casefold() for part in parts]

    return sorted(values, key=key)


def safe_read_image(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Read an image with OpenCV and check its dimensions and contents."""

    image_path = Path(path).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Input image path is not a file: {image_path}")

    image = cv2.imread(str(image_path), flags)
    if image is None:
        raise ValueError(f"OpenCV could not decode input image: {image_path}")
    if image.size == 0:
        raise ValueError(f"Input image is empty: {image_path}")
    return image


def read_grayscale(path: str | Path) -> np.ndarray:
    """Read a grayscale image at its native bit depth.

    Colour channels are converted to grayscale while source intensity units
    are preserved for thresholding.
    """

    image = safe_read_image(path, cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[..., 0]
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image, received shape {image.shape}")
    return np.asarray(image)


def list_image_files(directory: str | Path) -> list[Path]:
    """List common image files in a directory using natural filename order."""

    image_directory = Path(directory).expanduser()
    if not image_directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_directory}")
    if not image_directory.is_dir():
        raise NotADirectoryError(f"Image source is not a directory: {image_directory}")
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    return natural_sort(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.casefold() in suffixes
    )


def binarize_image(
    image: np.ndarray,
    *,
    threshold: float = 0.0,
    max_value: int = 255,
) -> np.ndarray:
    """Threshold an image and return a uint8 binary array."""

    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("Cannot binarize an empty image")
    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[..., 0]
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D image, received shape {array.shape}")

    finite = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    binary = np.where(finite > threshold, max_value, 0)
    return np.asarray(binary, dtype=np.uint8)


def binary_mask(image: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Return a uint8 mask for non-zero pixels or a supplied threshold."""

    if threshold is None:
        return (np.asarray(image) != 0).astype(np.uint8)
    return binarize_image(image, threshold=threshold, max_value=255)


def safe_divide(
    numerator: Any,
    denominator: Any,
    *,
    fill_value: float = 0.0,
    default: float | None = None,
) -> Any:
    """Divide arrays safely, replacing zero-denominator results.

    Scalar inputs produce a scalar result; array inputs produce a NumPy array
    with the broadcast shape of the operands.
    """

    if default is not None:
        fill_value = default
    left, right = np.broadcast_arrays(np.asarray(numerator), np.asarray(denominator))
    dtype = np.result_type(left.dtype, right.dtype, np.float64)
    result = np.full(left.shape, fill_value, dtype=dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(left, right, out=result, where=right != 0)
    result = np.where(np.isfinite(result), result, fill_value)
    if result.ndim == 0:
        return result.item()
    return result


def json_safe(value: Any) -> Any:
    """Recursively convert common scientific Python values to JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def write_json(path: str | Path, value: Any) -> Path:
    """Atomically write JSON with deterministic, human-readable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                relative_metadata(json_safe(value)),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path
