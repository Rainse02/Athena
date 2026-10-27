"""Exposure-normalized Gaussian defect density maps."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common import (
    binary_mask,
    list_image_files,
    read_grayscale,
    safe_divide,
    write_json,
)


_NATURAL_PARTS = re.compile(r"(\d+)")


def _natural_key(value: Any) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_PARTS.split(str(value))
    )


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _first(config: Any, general: Any, *names: str, default: Any = None) -> Any:
    value = _get(config, *names, default=None)
    return value if value is not None else _get(general, *names, default=default)


def _paths(source: Any) -> list[Path]:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            paths = [Path(item) for item in list_image_files(path)]
        elif path.is_file():
            paths = [path]
        else:
            raise FileNotFoundError(path)
    else:
        paths = [Path(item) for item in source]
    paths = sorted(paths, key=_natural_key)
    if not paths:
        raise ValueError("image source is empty")
    return paths


def _read_2d(path: Path) -> np.ndarray:
    try:
        image = np.asarray(read_grayscale(path))
    except Exception as exc:
        raise ValueError(f"could not read image {path}") from exc
    if image.ndim != 2:
        raise ValueError(f"image {path} is not 2-D")
    return image


def _mask(image: np.ndarray) -> np.ndarray:
    converted = np.asarray(binary_mask(image), dtype=bool)
    nonzero = np.asarray(image) != 0
    if converted.shape != nonzero.shape or np.any(nonzero & ~converted):
        return nonzero
    return converted


def _source(config: Any, general: Any) -> Any:
    value = _first(
        config,
        general,
        "defect_path",
        "defect_image",
        "image_path",
        "input_path",
        "path",
        "defect_dir",
    )
    if value is None:
        raise ValueError("no density input image was configured")
    return value


def _exposure(config: Any, general: Any, shape: tuple[int, int]) -> np.ndarray:
    value = _first(
        config,
        general,
        "exposure_mask",
        "exposure_path",
        "exposure",
        "valid_mask",
        "mask_path",
    )
    if value is None:
        return np.ones(shape, dtype=np.float32)
    if isinstance(value, np.ndarray):
        array = value
    else:
        paths = _paths(value)
        if len(paths) != 1:
            raise ValueError("density exposure must be one 2-D mask")
        array = _read_2d(paths[0])
    if tuple(array.shape) != shape:
        raise ValueError(f"exposure mask shape {array.shape} does not match {shape}")
    return (np.asarray(array) != 0).astype(np.float32)


def _sigma_values(config: Any, general: Any) -> list[float]:
    values = _first(
        config,
        general,
        "sigmas_mm",
        "sigma_mm",
        "gaussian_sigmas_mm",
        default=(0.5, 1.0, 2.0),
    )
    if np.isscalar(values):
        values = [values]
    values = [float(value) for value in values]
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Gaussian sigma values must be finite and positive")
    return values


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_json(path, dict(value))


def _save_float_tiff(path: Path, array: np.ndarray) -> None:
    try:
        import tifffile

        tifffile.imwrite(
            path,
            np.asarray(array, dtype=np.float32),
            compression="zlib",
            compressionargs={"level": 6},
            tile=(256, 256),
        )
    except ImportError as exc:
        raise RuntimeError("tifffile is required to write density TIFF files") from exc


def _plot_density(
    path: Path,
    density: np.ndarray,
    sigma_mm: float,
    pixel_size_mm: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data_ratio = density.shape[0] / max(1, density.shape[1])
    if data_ratio >= 1.0:
        width = 4.8
        height = min(14.0, width * data_ratio)
    else:
        height = 4.8
        width = min(14.0, height / max(data_ratio, np.finfo(float).eps))
    figure, axis = plt.subplots(figsize=(width, height))
    masked = np.ma.masked_invalid(density)
    extent = [
        0.0,
        density.shape[1] * pixel_size_mm,
        density.shape[0] * pixel_size_mm,
        0.0,
    ]
    image = axis.imshow(
        masked,
        cmap="viridis",
        interpolation="nearest",
        extent=extent,
        aspect="equal",
    )
    axis.set_xlabel("Flattened x position (mm)")
    axis.set_ylabel("Flattened y position (mm)")
    axis.set_title(f"Exposure-normalized density, σ={sigma_mm:g} mm")
    figure.colorbar(image, ax=axis, label="defect / exposed pixel")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_density_analysis(
    config: Any, general: Any, output_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Create normalized Gaussian density maps at one or more physical scales.

    The numerator and denominator are convolved with the same Gaussian and
    divided pointwise.  Consequently a uniform defect field remains uniform at
    image boundaries, where an unnormalized blur would otherwise be biased.
    Pixels with zero exposure are represented by NaN and counted in the
    summary.
    """

    source = _source(config, general)
    paths = _paths(source)
    if len(paths) != 1:
        raise ValueError("density analysis expects one 2-D defect image")
    defect_image = _read_2d(paths[0])
    defect = _mask(defect_image).astype(np.float32)
    exposure = _exposure(config, general, tuple(defect.shape))
    pixel_size = _first(
        config, general, "pixel_size_mm", "pixel_spacing_mm", "spacing_mm", default=1.0
    )
    pixel_size = float(pixel_size)
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel size must be finite and positive")
    sigmas = _sigma_values(config, general)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise RuntimeError("scipy is required for Gaussian density analysis") from exc

    truncate = float(_first(config, general, "gaussian_truncate", default=4.0))
    if not np.isfinite(truncate) or truncate <= 0:
        raise ValueError("gaussian_truncate must be finite and positive")
    defect_within_exposure = defect * exposure
    outputs: dict[str, dict[str, Any]] = {}
    for sigma_mm in sigmas:
        sigma_px = sigma_mm / pixel_size
        numerator = gaussian_filter(
            defect_within_exposure,
            sigma=sigma_px,
            mode="constant",
            cval=0.0,
            truncate=truncate,
        )
        denominator = gaussian_filter(
            exposure,
            sigma=sigma_px,
            mode="constant",
            cval=0.0,
            truncate=truncate,
        )
        density = np.asarray(
            safe_divide(numerator, denominator, fill_value=np.nan),
            dtype=np.float32,
        )
        token = f"{sigma_mm:g}".replace(".", "p")
        tiff_path = output / f"density_{token}mm.tif"
        png_path = output / f"density_{token}mm.png"
        _save_float_tiff(tiff_path, density)
        _plot_density(png_path, density, sigma_mm, pixel_size)
        finite = density[np.isfinite(density)]
        outputs[str(sigma_mm)] = {
            "sigma_px": sigma_px,
            "tiff": str(tiff_path),
            "png": str(png_path),
            "finite_pixel_count": int(finite.size),
            "minimum": float(np.min(finite)) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "mean": float(np.mean(finite)) if finite.size else None,
            "p95": float(np.percentile(finite, 95)) if finite.size else None,
            "p99": float(np.percentile(finite, 99)) if finite.size else None,
            "maximum": float(np.max(finite)) if finite.size else None,
        }
    summary = {
        "analysis": "density",
        "shape": [int(value) for value in defect.shape],
        "pixel_size_mm": pixel_size,
        "sigmas_mm": sigmas,
        "gaussian_truncate": truncate,
        "normalization": "Gaussian(defect within exposure) / Gaussian(exposure)",
        "raw_defect_pixels_within_exposure": int(
            np.count_nonzero(defect_within_exposure)
        ),
        "raw_exposed_pixels": int(np.count_nonzero(exposure)),
        "raw_defect_fraction": (
            float(np.count_nonzero(defect_within_exposure) / np.count_nonzero(exposure))
            if np.count_nonzero(exposure)
            else None
        ),
        "tiff_storage": "float32, tiled 256 x 256, zlib level 6",
        "raw_zero_exposure_pixels": int(np.count_nonzero(exposure == 0)),
        "zero_exposure_pixels": int(np.count_nonzero(exposure == 0)),
        "files": outputs,
    }
    _write_json(output / "density_summary.json", summary)
    return {**summary, "n_scales": int(len(sigmas))}


__all__ = ["run_density_analysis"]
