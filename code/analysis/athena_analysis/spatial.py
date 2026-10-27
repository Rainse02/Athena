"""Spatial summaries for segmented electrode defects.

The functions in this module deliberately keep the image stack streaming.  A
stack is reduced one slice at a time and no result depends on the grey level
being 255: every non-zero input pixel is a defect pixel.
"""

from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .common import (
    binary_mask,
    list_image_files,
    read_grayscale,
    safe_divide,
    write_json,
)


_NATURAL_PARTS = re.compile(r"(\d+)")


def _natural_key(value: Any) -> tuple[Any, ...]:
    """Sort names as humans expect (``slice2`` before ``slice10``)."""

    text = str(value)
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_PARTS.split(text)
    )


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a value from either a mapping or a small configuration object."""

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
    if value is not None:
        return value
    return _get(general, *names, default=default)


def _source(config: Any, general: Any, names: Sequence[str]) -> Any:
    source = _first(config, general, *names)
    if source is not None:
        return source
    raise ValueError(
        "no image source was configured; expected one of " + ", ".join(names)
    )


def _paths(source: Any) -> list[Path]:
    """Resolve a directory, a file, or an explicit sequence of image paths."""

    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            candidates = list_image_files(path)
            paths = [Path(item) for item in candidates]
        elif path.is_file():
            paths = [path]
        else:
            raise FileNotFoundError(path)
    else:
        try:
            paths = [Path(item) for item in source]
        except TypeError as exc:
            raise TypeError(
                "image source must be a path or a sequence of paths"
            ) from exc
    paths = sorted(paths, key=_natural_key)
    if not paths:
        raise ValueError(f"image source is empty: {source!r}")
    return paths


def _read_2d(path: Path) -> np.ndarray:
    try:
        image = np.asarray(read_grayscale(path))
    except Exception as exc:
        raise ValueError(f"could not read image {path}") from exc
    if image.ndim != 2:
        raise ValueError(f"image {path} is not a 2-D grayscale image")
    return image


def _defect_bool(image: np.ndarray) -> np.ndarray:
    """Convert any non-zero image value to bool through the shared API."""

    # ``binary_mask`` is the project-wide conversion hook.  The explicit
    # non-zero fallback guarantees that a valid, non-255 label is retained.
    converted = np.asarray(binary_mask(image), dtype=bool)
    nonzero = np.asarray(image) != 0
    if converted.shape != nonzero.shape or np.any(nonzero & ~converted):
        return nonzero
    return converted


def _iter_stack(
    source: Any, limit: int | None = None
) -> Iterator[tuple[int, Path, np.ndarray]]:
    paths = _paths(source)
    if limit is not None:
        if limit <= 0:
            raise ValueError("slice_limit must be positive")
        paths = paths[:limit]
    if not paths:
        raise ValueError("no slices remain after applying slice_limit")
    expected: tuple[int, int] | None = None
    for index, path in enumerate(paths):
        defect = _defect_bool(_read_2d(path))
        shape = tuple(defect.shape)
        if expected is None:
            expected = shape
        elif shape != expected:
            raise ValueError(
                f"inconsistent image shape at {path}: expected {expected}, got {shape}"
            )
        yield index, path, defect


def _array_mask(value: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or tuple(array.shape) != shape:
        raise ValueError(f"exposure mask shape {array.shape} does not match {shape}")
    return array != 0


def _exposure_source(config: Any, general: Any) -> Any:
    return _first(
        config,
        general,
        "exposure_mask",
        "exposure_path",
        "exposure_dir",
        "exposure",
        "valid_mask",
        "mask_path",
    )


def _exposure_for_slice(
    source: Any,
    index: int,
    shape: tuple[int, int],
    cache: dict[str, Any],
) -> np.ndarray:
    if source is None:
        return np.ones(shape, dtype=bool)
    if isinstance(source, np.ndarray):
        return _array_mask(source, shape)
    if (
        isinstance(source, (list, tuple))
        and source
        and not isinstance(source[0], (int, float))
    ):
        paths = cache.setdefault(
            "paths", sorted([Path(item) for item in source], key=_natural_key)
        )
        if len(paths) == 1:
            return _array_mask(_read_2d(paths[0]), shape)
        if index >= len(paths):
            raise ValueError("exposure stack has fewer slices than defect stack")
        return _array_mask(_read_2d(paths[index]), shape)
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            paths = cache.setdefault("paths", _paths(path))
            if len(paths) == 1:
                return _array_mask(_read_2d(paths[0]), shape)
            if index >= len(paths):
                raise ValueError("exposure stack has fewer slices than defect stack")
            return _array_mask(_read_2d(paths[index]), shape)
        return _array_mask(_read_2d(path), shape)
    return _array_mask(source, shape)


def _pixel_size(config: Any, general: Any, *, collector: bool = False) -> float:
    names = (
        ("collector_pixel_size_mm", "collector_spacing_mm")
        if collector
        else ("pixel_size_mm", "pixel_spacing_mm", "spacing_mm", "resolution_mm")
    )
    value = _first(config, general, *names, default=1.0)
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0]
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("pixel spacing must be a finite positive number")
    return value


def _center(config: Any, general: Any, shape: tuple[int, int]) -> tuple[float, float]:
    value = _first(config, general, "center", "image_center")
    if value is not None:
        if isinstance(value, Mapping):
            x = value.get("x", value.get("cx"))
            y = value.get("y", value.get("cy"))
        else:
            x, y = value
        return float(x), float(y)
    x = _first(config, general, "center_x", "cx", default=(shape[1] - 1) / 2.0)
    y = _first(config, general, "center_y", "cy", default=(shape[0] - 1) / 2.0)
    return float(x), float(y)


def _radius_limits(
    config: Any, general: Any, pixel_size: float, max_radius: float
) -> tuple[float, float]:
    range_px = _first(config, general, "radius_range_px", "radial_range_px")
    if range_px is not None:
        minimum, maximum = map(float, range_px)
    else:
        range_mm = _first(config, general, "radius_range_mm", "radial_range_mm")
        if range_mm is not None:
            minimum, maximum = (float(v) / pixel_size for v in range_mm)
        else:
            minimum = float(
                _first(
                    config,
                    general,
                    "radius_min_px",
                    "radial_min_px",
                    "r_min_px",
                    default=0.0,
                )
            )
            maximum = float(
                _first(
                    config,
                    general,
                    "radius_max_px",
                    "radial_max_px",
                    "max_radius_px",
                    "r_max_px",
                    default=max_radius,
                )
            )
    if not (0 <= minimum < maximum):
        raise ValueError("radius range must satisfy 0 <= minimum < maximum")
    clipped_maximum = min(maximum, max_radius)
    if minimum >= clipped_maximum:
        raise ValueError("radius range does not intersect the image")
    return minimum, clipped_maximum


def _edges(
    config: Any,
    general: Any,
    prefix: str,
    minimum: float,
    maximum: float,
    pixel_size: float,
) -> np.ndarray:
    explicit = _first(
        config,
        general,
        f"{prefix}_bins",
        f"{prefix}_edges",
        f"{prefix}_bins_px",
        f"{prefix}_edges_px",
    )
    if explicit is not None:
        if np.isscalar(explicit):
            count = int(explicit)
            if count <= 0:
                raise ValueError(f"{prefix}_bins must be positive")
            return np.linspace(minimum, maximum, count + 1)
        edges = np.asarray(explicit, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(f"{prefix}_bins must contain at least two edges")
        return edges
    width = _first(config, general, f"{prefix}_bin_width_px", "bin_width_px")
    if width is None:
        width_mm = _first(config, general, f"{prefix}_bin_width_mm", "bin_width_mm")
        width = None if width_mm is None else float(width_mm) / pixel_size
    if width is None:
        count = int(
            _first(
                config,
                general,
                f"n_{prefix}_bins",
                default=max(1, int(math.ceil(maximum - minimum))),
            )
        )
        return np.linspace(minimum, maximum, count + 1)
    width = float(width)
    if width <= 0:
        raise ValueError(f"{prefix} bin width must be positive")
    values = np.arange(minimum, maximum, width, dtype=float)
    edges = np.concatenate([values, [maximum]])
    return np.unique(edges)


def _moving_block_ci(
    counts: np.ndarray,
    exposure: np.ndarray,
    block: int,
    seed: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Moving-block bootstrap of a ratio using overlapping finite blocks.

    The z stack is not periodic, so sampled blocks never wrap from the final
    slice to the first slice.
    """

    n_slices, n_bins = counts.shape
    if n_slices == 0 or samples <= 0:
        nan = np.full(n_bins, np.nan, dtype=float)
        return nan, nan.copy()
    block = max(1, min(int(block), n_slices))
    rng = np.random.default_rng(seed)
    boot = np.empty((samples, n_bins), dtype=float)
    blocks_per_sample = int(math.ceil(n_slices / block))
    within_block = np.arange(block, dtype=int)
    for sample_index in range(samples):
        starts = rng.integers(
            0,
            n_slices - block + 1,
            size=blocks_per_sample,
        )
        selected = (starts[:, None] + within_block[None, :]).ravel()[:n_slices]
        boot[sample_index] = safe_divide(
            counts[selected].sum(axis=0),
            exposure[selected].sum(axis=0),
            fill_value=np.nan,
        )
    return np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _write_slice_counts_csv(
    path: Path,
    counts: np.ndarray,
    exposure: np.ndarray,
    slice_names: Sequence[str],
    slice_spacing_mm: float,
) -> None:
    """Write per-slice values incrementally to avoid a large row list."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "slice_index",
                "slice_file",
                "slice_position_mm",
                "bin",
                "raw_count",
                "exposure",
                "density",
            )
        )
        for slice_index in range(counts.shape[0]):
            for bin_index in range(counts.shape[1]):
                writer.writerow(
                    (
                        slice_index,
                        slice_names[slice_index],
                        slice_index * slice_spacing_mm,
                        bin_index,
                        counts[slice_index, bin_index],
                        exposure[slice_index, bin_index],
                        safe_divide(
                            counts[slice_index, bin_index],
                            exposure[slice_index, bin_index],
                            fill_value=np.nan,
                        ),
                    )
                )


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(summary))


def _plot(
    path_stem: Path,
    x: np.ndarray,
    y: np.ndarray,
    low: np.ndarray | None,
    high: np.ndarray | None,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(x, y, color="#245a7a", linewidth=1.6)
    if low is not None and high is not None:
        axis.fill_between(x, low, high, color="#8bb8cf", alpha=0.35, linewidth=0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path_stem.with_suffix(".png"), dpi=180)
    figure.savefig(path_stem.with_suffix(".svg"))
    plt.close(figure)


def _plot_profiles(
    path_stem: Path,
    x: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    xlabel: str,
    ylabel: str,
    first_label: str,
    second_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(x, first, color="#245a7a", linewidth=1.5, label=first_label)
    axis.plot(x, second, color="#c35a3a", linewidth=1.5, label=second_label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path_stem.with_suffix(".png"), dpi=180)
    figure.savefig(path_stem.with_suffix(".svg"))
    plt.close(figure)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: Sequence[float],
) -> list[float | None]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return [None for _ in probabilities]
    order = np.argsort(values[valid], kind="mergesort")
    sorted_values = values[valid][order]
    sorted_weights = weights[valid][order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= np.sum(sorted_weights)
    return [
        float(np.interp(probability, cumulative, sorted_values))
        for probability in probabilities
    ]


def _common_analysis_options(config: Any, general: Any) -> tuple[int, int, int]:
    block = int(_first(config, general, "bootstrap_block", "block_size", default=20))
    samples = int(
        _first(config, general, "bootstrap_samples", "n_bootstrap", default=1000)
    )
    seed = int(_first(config, general, "bootstrap_seed", "seed", default=20260819))
    return block, samples, seed


def run_radial_analysis(
    config: Any, general: Any, output_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Compute radial defect density using the number of exposed pixels per ring."""

    source = _source(
        config,
        general,
        ("defect_dir", "defect_path", "image_dir", "input_dir", "images", "path"),
    )
    exposure_source = _exposure_source(config, general)
    pixel_size = _pixel_size(config, general)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    first_shape: tuple[int, int] | None = None
    x_center = y_center = 0.0
    radii: np.ndarray | None = None
    counts_by_slice: list[np.ndarray] = []
    exposure_by_slice: list[np.ndarray] = []
    slice_names: list[str] = []
    raw_total: np.ndarray | None = None
    exposure_total: np.ndarray | None = None
    exposure_cache: dict[str, Any] = {}
    membership: np.ndarray | None = None
    radial_bin_index: np.ndarray | None = None
    for index, slice_path, defect in _iter_stack(source):
        if first_shape is None:
            first_shape = tuple(defect.shape)
            x_center, y_center = _center(config, general, first_shape)
            max_radius = float(
                np.hypot(
                    max(x_center, first_shape[1] - 1 - x_center),
                    max(y_center, first_shape[0] - 1 - y_center),
                )
            )
            minimum, maximum = _radius_limits(config, general, pixel_size, max_radius)
            radii = _edges(config, general, "radial", minimum, maximum, pixel_size)
            raw_total = np.zeros(len(radii) - 1, dtype=float)
            exposure_total = np.zeros(len(radii) - 1, dtype=float)
            yy, xx = np.indices(first_shape, dtype=float)
            radius = np.hypot(xx - x_center, yy - y_center)
            membership = (radius >= radii[0]) & (radius < radii[-1])
            radial_bin_index = np.digitize(radius[membership], radii) - 1
        exposure = _exposure_for_slice(
            exposure_source, index, first_shape, exposure_cache
        )
        exposed = exposure[membership]
        defects = defect[membership]
        slice_counts = np.bincount(
            radial_bin_index[exposed],
            weights=defects[exposed],
            minlength=len(radii) - 1,
        ).astype(float)
        slice_exposure = np.bincount(
            radial_bin_index,
            weights=exposed,
            minlength=len(radii) - 1,
        ).astype(float)
        counts_by_slice.append(slice_counts)
        exposure_by_slice.append(slice_exposure)
        slice_names.append(slice_path.name)
        raw_total += slice_counts
        exposure_total += slice_exposure

    counts_array = np.asarray(counts_by_slice)
    exposure_array = np.asarray(exposure_by_slice)
    density = safe_divide(raw_total, exposure_total, fill_value=np.nan)
    block, samples, seed = _common_analysis_options(config, general)
    ci_low, ci_high = _moving_block_ci(
        counts_array, exposure_array, block, seed, samples
    )
    radius_px = (radii[:-1] + radii[1:]) / 2.0
    radius_mm = radius_px * pixel_size
    rows = []
    for idx in range(len(radius_px)):
        rows.append(
            {
                "bin": idx,
                "radius_px": radius_px[idx],
                "radius_mm": radius_mm[idx],
                "raw_count": raw_total[idx],
                "exposure": exposure_total[idx],
                "density": density[idx],
                "ci95_low": ci_low[idx],
                "ci95_high": ci_high[idx],
            }
        )
    csv_path = output / "radial.csv"
    _write_csv(csv_path, rows, tuple(rows[0].keys()))
    slice_path = output / "radial_slice_counts.csv"
    slice_spacing = float(
        _first(config, general, "slice_spacing_mm", default=pixel_size)
    )
    _write_slice_counts_csv(
        slice_path,
        counts_array,
        exposure_array,
        slice_names,
        slice_spacing,
    )
    _plot(
        output / "radial",
        radius_mm,
        density,
        ci_low,
        ci_high,
        "Radius from center (mm)",
        "Defect probability per exposed pixel",
    )
    radius_quantiles = _weighted_quantile(
        radius_mm,
        raw_total,
        (0.1, 0.25, 0.5, 0.75, 0.9),
    )
    summary = {
        "analysis": "radial",
        "n_slices": int(len(counts_array)),
        "pixel_size_mm": pixel_size,
        "radius_range_px": [float(radii[0]), float(radii[-1])],
        "exposure_mode": (
            "configured mask"
            if exposure_source is not None
            else "geometric ring pixels"
        ),
        "bootstrap": {"samples": samples, "block_length_slices": block, "seed": seed},
        "ci_interpretation": "The interval is a within-cell moving-block spatial bootstrap over z slices; it is not a between-cell population confidence interval.",
        "files": {
            "csv": str(csv_path),
            "slice_counts_csv": str(slice_path),
            "png": str(output / "radial.png"),
            "svg": str(output / "radial.svg"),
        },
        "zero_exposure_bins": int(np.count_nonzero(exposure_total == 0)),
        "global_defect_fraction": float(
            safe_divide(np.sum(raw_total), np.sum(exposure_total), fill_value=np.nan)
        ),
        "defect_weighted_radius_mm": {
            "mean": float(
                safe_divide(
                    np.sum(radius_mm * raw_total),
                    np.sum(raw_total),
                    fill_value=np.nan,
                )
            ),
            "q10": radius_quantiles[0],
            "q25": radius_quantiles[1],
            "median": radius_quantiles[2],
            "q75": radius_quantiles[3],
            "q90": radius_quantiles[4],
        },
    }
    _write_summary(output / "radial_summary.json", summary)
    return {
        **summary,
        "n_bins": int(len(radius_px)),
        "total_raw_count": float(np.sum(raw_total)),
        "total_exposure": float(np.sum(exposure_total)),
    }


def run_angle_analysis(
    config: Any, general: Any, output_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Compute angular defect density normalized by the exposed pixels per sector."""

    source = _source(
        config,
        general,
        ("defect_dir", "defect_path", "image_dir", "input_dir", "images", "path"),
    )
    exposure_source = _exposure_source(config, general)
    pixel_size = _pixel_size(config, general)
    slice_limit = _first(config, general, "slice_limit", default=None)
    if slice_limit is not None:
        slice_limit = int(slice_limit)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts_by_slice: list[np.ndarray] = []
    exposure_by_slice: list[np.ndarray] = []
    slice_names: list[str] = []
    exposure_cache: dict[str, Any] = {}
    edges: np.ndarray | None = None
    raw_total = exposure_total = None
    for index, slice_file, defect in _iter_stack(source, slice_limit):
        if edges is None:
            shape = tuple(defect.shape)
            cx, cy = _center(config, general, shape)
            max_radius = float(
                np.hypot(max(cx, shape[1] - 1 - cx), max(cy, shape[0] - 1 - cy))
            )
            minimum, maximum = _radius_limits(config, general, pixel_size, max_radius)
            angle_width = _first(
                config,
                general,
                "angle_bin_width_deg",
                "bin_width_deg",
                default=None,
            )
            explicit = _first(config, general, "angle_bins", "angle_edges")
            if explicit is not None:
                if np.isscalar(explicit):
                    edges = np.linspace(-180.0, 180.0, int(explicit) + 1)
                else:
                    edges = np.asarray(explicit, dtype=float)
            else:
                count = int(_first(config, general, "n_angle_bins", default=360))
                if angle_width is not None:
                    count = max(1, int(math.ceil(360.0 / float(angle_width))))
                edges = np.linspace(-180.0, 180.0, count + 1)
            yy, xx = np.indices(shape, dtype=float)
            radius = np.hypot(xx - cx, yy - cy)
            angle = np.degrees(np.arctan2(cy - yy, xx - cx))
            # Keep the negative-x ray in the periodic domain instead of
            # dropping exact +180-degree pixels at the final bin edge.
            angle = (angle + 180.0) % 360.0 - 180.0
            radial_membership = (radius >= minimum) & (radius < maximum)
            raw_total = np.zeros(len(edges) - 1, dtype=float)
            exposure_total = np.zeros(len(edges) - 1, dtype=float)
        exposure = _exposure_for_slice(exposure_source, index, shape, exposure_cache)
        selected = radial_membership & exposure
        bins = np.digitize(angle[selected], edges, right=False) - 1
        valid = (bins >= 0) & (bins < len(edges) - 1)
        slice_counts = np.bincount(
            bins[valid], weights=defect[selected][valid], minlength=len(edges) - 1
        ).astype(float)
        slice_exposure = np.bincount(
            bins[valid],
            weights=np.ones(np.count_nonzero(valid)),
            minlength=len(edges) - 1,
        ).astype(float)
        counts_by_slice.append(slice_counts)
        exposure_by_slice.append(slice_exposure)
        slice_names.append(slice_file.name)
        raw_total += slice_counts
        exposure_total += slice_exposure
    counts_array = np.asarray(counts_by_slice)
    exposure_array = np.asarray(exposure_by_slice)
    density = safe_divide(raw_total, exposure_total, fill_value=np.nan)
    block, samples, seed = _common_analysis_options(config, general)
    ci_low, ci_high = _moving_block_ci(
        counts_array, exposure_array, block, seed, samples
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    rows = []
    for idx in range(len(centers)):
        rows.append(
            {
                "bin": idx,
                "angle_deg": centers[idx],
                "raw_count": raw_total[idx],
                "exposure": exposure_total[idx],
                "density": density[idx],
                "ci95_low": ci_low[idx],
                "ci95_high": ci_high[idx],
            }
        )
    csv_path = output / "angle.csv"
    _write_csv(csv_path, rows, tuple(rows[0].keys()))
    slice_path = output / "angle_slice_counts.csv"
    slice_spacing = float(
        _first(config, general, "slice_spacing_mm", default=pixel_size)
    )
    _write_slice_counts_csv(
        slice_path,
        counts_array,
        exposure_array,
        slice_names,
        slice_spacing,
    )
    _plot(
        output / "angle",
        centers,
        density,
        ci_low,
        ci_high,
        "Angle from +x (degree)",
        "Defect probability per exposed pixel",
    )
    finite_density = np.where(np.isfinite(density), density, 0.0)
    angle_radians = np.deg2rad(centers)
    density_sum = float(np.sum(finite_density))
    first_resultant = (
        np.sum(finite_density * np.exp(1.0j * angle_radians)) / density_sum
        if density_sum > 0
        else complex(np.nan, np.nan)
    )
    second_resultant = (
        np.sum(finite_density * np.exp(2.0j * angle_radians)) / density_sum
        if density_sum > 0
        else complex(np.nan, np.nan)
    )
    peak_index = int(np.nanargmax(density)) if np.any(np.isfinite(density)) else None
    summary = {
        "analysis": "angle",
        "n_slices": int(len(counts_array)),
        "slice_limit": slice_limit,
        "first_slice_file": slice_names[0],
        "last_slice_file": slice_names[-1],
        "center_px": [float(cx), float(cy)],
        "radius_range_px_half_open": [float(minimum), float(maximum)],
        "angle_domain_deg_half_open": [-180.0, 180.0],
        "exposure_mode": (
            "configured mask"
            if exposure_source is not None
            else "geometric annulus pixels"
        ),
        "bootstrap": {"samples": samples, "block_length_slices": block, "seed": seed},
        "ci_interpretation": "The interval is a within-cell moving-block spatial bootstrap over z slices; it is not a between-cell population confidence interval.",
        "files": {
            "csv": str(csv_path),
            "slice_counts_csv": str(slice_path),
            "png": str(output / "angle.png"),
            "svg": str(output / "angle.svg"),
        },
        "zero_exposure_bins": int(np.count_nonzero(exposure_total == 0)),
        "global_defect_fraction": float(
            safe_divide(np.sum(raw_total), np.sum(exposure_total), fill_value=np.nan)
        ),
        "circular_summary_descriptive_only": {
            "first_harmonic_resultant_length": float(abs(first_resultant)),
            "preferred_direction_deg": (
                float(np.rad2deg(np.angle(first_resultant)))
                if np.isfinite(first_resultant.real)
                else None
            ),
            "second_harmonic_resultant_length": float(abs(second_resultant)),
            "preferred_axis_deg_mod_180": (
                float((np.rad2deg(np.angle(second_resultant)) / 2.0) % 180.0)
                if np.isfinite(second_resultant.real)
                else None
            ),
            "angular_rate_coefficient_of_variation": (
                float(np.std(finite_density) / np.mean(finite_density))
                if np.mean(finite_density) > 0
                else None
            ),
            "peak_angle_deg": (
                float(centers[peak_index]) if peak_index is not None else None
            ),
            "interpretation": "These coefficients summarize within-cell anisotropy. No independent-cell inferential p-value is reported.",
        },
    }
    _write_summary(output / "angle_summary.json", summary)
    return {
        **summary,
        "n_bins": int(len(centers)),
        "total_raw_count": float(np.sum(raw_total)),
        "total_exposure": float(np.sum(exposure_total)),
    }


def _corr(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if second.size != first.size:
        return float("nan")
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _corr_lag(first: np.ndarray, second: np.ndarray, lag: int) -> float:
    if lag > 0:
        return _corr(first[:-lag], second[lag:])
    if lag < 0:
        return _corr(first[-lag:], second[:lag])
    return _corr(first, second)


def _collector_rows(source: Any) -> np.ndarray:
    if isinstance(source, np.ndarray):
        image = np.asarray(source)
    else:
        paths = _paths(source)
        if len(paths) != 1:
            raise ValueError("collector source must be one image")
        image = _read_2d(paths[0])
    if image.ndim != 2:
        raise ValueError("collector image must be 2-D")
    return np.count_nonzero(image != 0, axis=1).astype(float) / float(image.shape[1])


def _resample(
    axis: np.ndarray, values: np.ndarray, common_axis: np.ndarray
) -> np.ndarray:
    return np.interp(common_axis, axis, values)


def run_vertical_analysis(
    config: Any, general: Any, output_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Compare slice-wise defect fractions with row-wise collector foreground.

    The p-value is obtained by repeating the complete lag search after random
    circular shifts.  It therefore tests the selected maximum, not a single
    pre-selected correlation.
    """

    defect_source = _source(
        config,
        general,
        ("defect_dir", "defect_path", "image_dir", "input_dir", "images", "path"),
    )
    collector_source = _source(
        config,
        general,
        (
            "collector_path",
            "collector_image",
            "collector_dir",
            "current_collector_path",
        ),
    )
    exposure_source = _exposure_source(config, general)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spacing = _first(
        config,
        general,
        "slice_spacing_mm",
        "z_spacing_mm",
        default=_pixel_size(config, general),
    )
    spacing = float(spacing)
    if spacing <= 0:
        raise ValueError("slice spacing must be positive")
    defect_values: list[float] = []
    defect_slice_names: list[str] = []
    exposure_cache: dict[str, Any] = {}
    geometric_radius = _first(config, general, "exposure_radius_px", default=None)
    geometric_exposure: np.ndarray | None = None
    for index, defect_path, defect in _iter_stack(defect_source):
        if exposure_source is None and geometric_radius is not None:
            if geometric_exposure is None:
                cx, cy = _center(config, general, tuple(defect.shape))
                yy, xx = np.indices(defect.shape, dtype=float)
                geometric_exposure = np.hypot(xx - cx, yy - cy) <= float(
                    geometric_radius
                )
            exposure = geometric_exposure
        else:
            exposure = _exposure_for_slice(
                exposure_source,
                index,
                tuple(defect.shape),
                exposure_cache,
            )
        exposed = int(np.count_nonzero(exposure))
        defect_values.append(
            float(np.count_nonzero(defect & exposure) / exposed)
            if exposed
            else float("nan")
        )
        defect_slice_names.append(defect_path.name)
    defect_values_array = np.asarray(defect_values, dtype=float)
    collector_values = _collector_rows(collector_source)
    collector_spacing = float(
        _first(
            config,
            general,
            "collector_spacing_mm",
            default=_pixel_size(config, general, collector=True),
        )
    )
    if collector_spacing <= 0:
        raise ValueError("collector spacing must be positive")
    max_axis = min(
        (len(defect_values_array) - 1) * spacing,
        (len(collector_values) - 1) * collector_spacing,
    )
    step = float(
        _first(
            config,
            general,
            "vertical_bin_mm",
            "resample_spacing_mm",
            default=min(spacing, collector_spacing),
        )
    )
    common_axis = np.arange(0.0, max_axis + step * 0.5, step)
    defect_axis = np.arange(len(defect_values_array), dtype=float) * spacing
    collector_axis = np.arange(len(collector_values), dtype=float) * collector_spacing
    defect_resampled = _resample(defect_axis, defect_values_array, common_axis)
    collector_resampled = _resample(collector_axis, collector_values, common_axis)
    finite = np.isfinite(defect_resampled) & np.isfinite(collector_resampled)
    if not np.any(finite):
        raise ValueError("vertical profiles contain no finite overlapping samples")
    zero_lag_pearson = _corr(defect_resampled[finite], collector_resampled[finite])
    zero_lag_spearman = _corr(
        _rank(defect_resampled[finite]), _rank(collector_resampled[finite])
    )
    lag_max_px = _first(config, general, "max_lag_px", "lag_limit_px", default=None)
    lag_max_mm = _first(config, general, "max_lag_mm", "lag_limit_mm", default=None)
    if lag_max_px is not None:
        lag_limit = int(abs(float(lag_max_px)))
    elif lag_max_mm is not None:
        lag_limit = int(abs(float(lag_max_mm)) / step)
    else:
        lag_limit = min(50, max(0, len(common_axis) // 4))
    lag_limit = min(lag_limit, max(0, len(common_axis) - 2))
    lags = np.arange(-lag_limit, lag_limit + 1, dtype=int)
    lag_correlations = np.asarray(
        [_corr_lag(defect_resampled, collector_resampled, int(lag)) for lag in lags]
    )
    finite_lag = np.isfinite(lag_correlations)
    if not np.any(finite_lag):
        raise ValueError("no valid lag correlation")
    objective = str(
        _first(config, general, "lag_objective", default="positive")
    ).lower()
    if objective not in {"positive", "absolute"}:
        raise ValueError("lag_objective must be 'positive' or 'absolute'")
    lag_scores = (
        np.abs(lag_correlations) if objective == "absolute" else lag_correlations
    )
    best_position = int(np.nanargmax(lag_scores))
    best_lag = int(lags[best_position])
    aligned_correlation = float(lag_correlations[best_position])
    observed_score = float(lag_scores[best_position])
    if best_lag > 0:
        aligned_axis = common_axis[best_lag:]
        aligned_defect = defect_resampled[:-best_lag]
        aligned_collector = collector_resampled[best_lag:]
    elif best_lag < 0:
        aligned_axis = common_axis[:best_lag]
        aligned_defect = defect_resampled[-best_lag:]
        aligned_collector = collector_resampled[:best_lag]
    else:
        aligned_axis = common_axis
        aligned_defect = defect_resampled
        aligned_collector = collector_resampled
    aligned_finite = np.isfinite(aligned_defect) & np.isfinite(aligned_collector)
    aligned_spearman = _corr(
        _rank(aligned_defect[aligned_finite]),
        _rank(aligned_collector[aligned_finite]),
    )
    permutation_setting = _first(
        config,
        general,
        "permutations",
        "n_permutations",
        default=1000,
    )
    seed = int(_first(config, general, "permutation_seed", "seed", default=20260819))
    rng = np.random.default_rng(seed)
    if isinstance(permutation_setting, str):
        if permutation_setting.casefold() != "all":
            raise ValueError("permutations must be a positive integer or 'all'")
        shifts = np.arange(1, len(common_axis), dtype=int)
        permutation_mode = "all non-zero circular shifts"
    else:
        n_requested = int(permutation_setting)
        if n_requested <= 0:
            raise ValueError("permutations must be positive")
        shifts = rng.integers(1, len(common_axis), size=n_requested, dtype=int)
        permutation_mode = "random circular shifts with replacement"
    null_max = np.full(len(shifts), np.nan, dtype=float)
    null_zero = np.full(len(shifts), np.nan, dtype=float)
    if len(shifts) and len(common_axis) > 2:
        base_defect = np.nan_to_num(defect_resampled, nan=np.nanmean(defect_resampled))
        base_collector = np.nan_to_num(
            collector_resampled, nan=np.nanmean(collector_resampled)
        )
        for perm_index, shift in enumerate(shifts):
            shifted = np.roll(base_collector, shift)
            null_zero[perm_index] = _corr(base_defect, shifted)
            perm_corr = np.asarray(
                [_corr_lag(base_defect, shifted, int(lag)) for lag in lags]
            )
            perm_scores = np.abs(perm_corr) if objective == "absolute" else perm_corr
            null_max[perm_index] = np.nanmax(perm_scores)
    valid_null = null_max[np.isfinite(null_max)]
    p_value = (
        float(
            (1 + np.count_nonzero(valid_null >= observed_score)) / (1 + len(valid_null))
        )
        if valid_null.size
        else float("nan")
    )
    valid_zero = null_zero[np.isfinite(null_zero)]
    zero_lag_p_value = (
        float(
            (1 + np.count_nonzero(valid_zero >= zero_lag_pearson))
            / (1 + len(valid_zero))
        )
        if valid_zero.size and np.isfinite(zero_lag_pearson)
        else float("nan")
    )
    raw_path = output / "vertical_raw.csv"
    raw_rows = [
        {
            "position_mm": common_axis[i],
            "defect_fraction": defect_resampled[i],
            "collector_foreground_fraction": collector_resampled[i],
        }
        for i in range(len(common_axis))
    ]
    _write_csv(
        raw_path,
        raw_rows,
        ("position_mm", "defect_fraction", "collector_foreground_fraction"),
    )
    source_path = output / "vertical_defect_slices.csv"
    source_rows = [
        {
            "slice_index": index,
            "slice_file": defect_slice_names[index],
            "position_mm": index * spacing,
            "defect_fraction": defect_values_array[index],
        }
        for index in range(len(defect_values_array))
    ]
    _write_csv(
        source_path,
        source_rows,
        ("slice_index", "slice_file", "position_mm", "defect_fraction"),
    )
    aligned_path = output / "vertical_aligned.csv"
    aligned_rows = [
        {
            "position_mm": aligned_axis[i],
            "defect_fraction": aligned_defect[i],
            "collector_foreground_fraction": aligned_collector[i],
        }
        for i in range(len(aligned_axis))
    ]
    _write_csv(
        aligned_path,
        aligned_rows,
        ("position_mm", "defect_fraction", "collector_foreground_fraction"),
    )
    lag_path = output / "vertical_lag_scan.csv"
    lag_rows = [
        {
            "lag_pixels": int(lag),
            "lag_mm": float(lag * step),
            "pearson_r": correlation,
            "selection_score": score,
        }
        for lag, correlation, score in zip(lags, lag_correlations, lag_scores)
    ]
    _write_csv(
        lag_path,
        lag_rows,
        ("lag_pixels", "lag_mm", "pearson_r", "selection_score"),
    )
    null_path = output / "vertical_null_max_scores.csv"
    null_rows = [
        {
            "permutation": index,
            "circular_shift_pixels": int(shifts[index]),
            "zero_lag_pearson": null_zero[index],
            "maximum_selection_score": value,
        }
        for index, value in enumerate(null_max)
    ]
    _write_csv(
        null_path,
        null_rows,
        (
            "permutation",
            "circular_shift_pixels",
            "zero_lag_pearson",
            "maximum_selection_score",
        ),
    )
    _plot_profiles(
        output / "vertical_raw",
        common_axis,
        defect_resampled,
        collector_resampled,
        "Position (mm)",
        "Fraction",
        "Defect",
        "Collector",
    )
    _plot_profiles(
        output / "vertical_aligned",
        aligned_axis,
        aligned_defect,
        aligned_collector,
        "Aligned position (mm)",
        "Fraction",
        "Defect",
        "Collector",
    )
    if exposure_source is not None:
        exposure_description = "configured mask"
    elif geometric_radius is not None:
        exposure_description = f"circular domain, radius={float(geometric_radius):g} px"
    else:
        exposure_description = "full image"
    summary = {
        "analysis": "vertical",
        "exposure_mode": exposure_description,
        "zero_lag_pearson": zero_lag_pearson,
        "zero_lag_spearman": zero_lag_spearman,
        "zero_lag_positive_association_p_value": zero_lag_p_value,
        "lag_objective": objective,
        "lag_sign_convention": "positive lag means the collector profile is shifted toward increasing position relative to the defect profile",
        "best_lag_pixels": best_lag,
        "best_lag_mm": best_lag * step,
        "aligned_pearson": aligned_correlation,
        "aligned_spearman": aligned_spearman,
        "search_lags_pixels": [int(lags[0]), int(lags[-1])],
        "max_correlation_search_p_value": p_value,
        "p_value_method": "circular-shift null with the complete lag search repeated for each permutation",
        "inference_scope": "Within-cell spatial association only; one cell does not support between-cell population inference.",
        "defect_slice_count": int(len(defect_values_array)),
        "collector_row_count": int(len(collector_values)),
        "overlap_sample_count": int(len(common_axis)),
        "n_permutations": int(len(valid_null)),
        "permutation_mode": permutation_mode,
        "permutation_seed": seed,
        "files": {
            "raw_csv": str(raw_path),
            "defect_slices_csv": str(source_path),
            "aligned_csv": str(aligned_path),
            "lag_scan_csv": str(lag_path),
            "null_max_scores_csv": str(null_path),
            "raw_png": str(output / "vertical_raw.png"),
            "aligned_png": str(output / "vertical_aligned.png"),
        },
    }
    _write_summary(output / "vertical_summary.json", summary)
    return {**summary, "n_samples": int(len(common_axis))}


def run_flatten_analysis(
    config: Any, general: Any, output_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Write column-wise defect fractions for a flattened binary image."""

    source = _source(
        config,
        general,
        (
            "flatten_path",
            "flatten_image",
            "defect_path",
            "defect_dir",
            "image_path",
            "path",
        ),
    )
    exposure_source = _exposure_source(config, general)
    sum_columns: np.ndarray | None = None
    exposure_columns: np.ndarray | None = None
    shape: tuple[int, int] | None = None
    n_images = 0
    exposure_cache: dict[str, Any] = {}
    for index, _, defect in _iter_stack(source):
        if shape is None:
            shape = tuple(defect.shape)
            sum_columns = np.zeros(shape[1], dtype=float)
            exposure_columns = np.zeros(shape[1], dtype=float)
        exposure = _exposure_for_slice(exposure_source, index, shape, exposure_cache)
        sum_columns += np.sum(defect & exposure, axis=0)
        exposure_columns += np.sum(exposure, axis=0)
        n_images += 1
    defect_fraction = safe_divide(sum_columns, exposure_columns, fill_value=np.nan)
    pixel_size = _pixel_size(config, general)
    x_px = np.arange(shape[1], dtype=float)
    x_mm = x_px * pixel_size
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "flatten.csv"
    rows = [
        {
            "column": int(i),
            "position_mm": x_mm[i],
            "defect_fraction": defect_fraction[i],
        }
        for i in range(shape[1])
    ]
    _write_csv(csv_path, rows, ("column", "position_mm", "defect_fraction"))
    _plot(
        output / "flatten_pixel",
        x_mm,
        defect_fraction,
        None,
        None,
        "Flattened position (mm)",
        "Per-column defect fraction",
    )
    bin_width_mm = float(_first(config, general, "bin_width_mm", default=1.0))
    if not np.isfinite(bin_width_mm) or bin_width_mm <= 0:
        raise ValueError("flatten bin_width_mm must be finite and positive")
    bin_width_px = max(1, int(round(bin_width_mm / pixel_size)))
    binned_rows: list[dict[str, Any]] = []
    for bin_index, start in enumerate(range(0, shape[1], bin_width_px)):
        stop = min(shape[1], start + bin_width_px)
        binned_rows.append(
            {
                "bin": bin_index,
                "start_column": start,
                "stop_column_exclusive": stop,
                "position_mm": ((start + stop - 1) / 2.0) * pixel_size,
                "defect_pixels": float(np.sum(sum_columns[start:stop])),
                "exposure_pixels": float(np.sum(exposure_columns[start:stop])),
                "defect_fraction": safe_divide(
                    np.sum(sum_columns[start:stop]),
                    np.sum(exposure_columns[start:stop]),
                    fill_value=np.nan,
                ),
            }
        )
    binned_path = output / "flatten_binned.csv"
    _write_csv(binned_path, binned_rows, tuple(binned_rows[0].keys()))
    binned_position = np.asarray([row["position_mm"] for row in binned_rows])
    binned_fraction = np.asarray([row["defect_fraction"] for row in binned_rows])
    _plot(
        output / "flatten",
        binned_position,
        binned_fraction,
        None,
        None,
        "Flattened position (mm)",
        f"Defect fraction per {bin_width_mm:g} mm bin",
    )
    peak_index = (
        int(np.nanargmax(binned_fraction))
        if np.any(np.isfinite(binned_fraction))
        else None
    )
    summary = {
        "analysis": "flatten",
        "n_images": n_images,
        "exposure_mode": (
            "configured mask" if exposure_source is not None else "full image"
        ),
        "zero_exposure_columns": int(np.count_nonzero(exposure_columns == 0)),
        "global_defect_fraction": float(
            safe_divide(
                np.sum(sum_columns),
                np.sum(exposure_columns),
                fill_value=np.nan,
            )
        ),
        "nonzero_defect_columns": int(np.count_nonzero(sum_columns)),
        "bin_width_mm_requested": bin_width_mm,
        "bin_width_pixels": bin_width_px,
        "bin_width_mm_effective": bin_width_px * pixel_size,
        "peak_binned_position_mm": (
            float(binned_position[peak_index]) if peak_index is not None else None
        ),
        "peak_binned_defect_fraction": (
            float(binned_fraction[peak_index]) if peak_index is not None else None
        ),
        "periodic_seam_note": "The first and last columns are adjacent on the unwrapped circumference; a peak at either boundary should be interpreted as one seam-adjacent region.",
        "files": {
            "csv": str(csv_path),
            "binned_csv": str(binned_path),
            "png": str(output / "flatten.png"),
            "svg": str(output / "flatten.svg"),
            "pixel_png": str(output / "flatten_pixel.png"),
            "pixel_svg": str(output / "flatten_pixel.svg"),
        },
    }
    _write_summary(output / "flatten_summary.json", summary)
    return {**summary, "n_columns": int(len(x_mm))}


__all__ = [
    "run_radial_analysis",
    "run_angle_analysis",
    "run_vertical_analysis",
    "run_flatten_analysis",
]
