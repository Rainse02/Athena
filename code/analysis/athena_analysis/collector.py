"""Current-collector intersection and exposure analysis.

The collector runner has no import-time work.  Every generated image and
table is written beneath the caller-provided output directory; the input
``Slices.tif`` is never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from .common import binary_mask, read_grayscale, safe_divide, write_json


def _setting(config: Any, general: Any, key: str, default: Any = None) -> Any:
    for source in (config, general):
        if source is None:
            continue
        if isinstance(source, Mapping) and key in source:
            return source[key]
        if hasattr(source, key):
            return getattr(source, key)
    return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _json_safe(payload)
    write_json(path, data)


def _as_uint8(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    if result.dtype == np.uint8:
        return result
    if result.size == 0:
        return np.zeros(result.shape, dtype=np.uint8)
    finite = np.nan_to_num(result.astype(float), nan=0.0, posinf=255.0, neginf=0.0)
    if finite.min() >= 0 and finite.max() <= 255:
        return finite.astype(np.uint8)
    minimum = float(finite.min())
    maximum = float(finite.max())
    if maximum <= minimum:
        return np.zeros(result.shape, dtype=np.uint8)
    scaled = (finite - minimum) * 255.0 / (maximum - minimum)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _call_binary_mask(image: np.ndarray, threshold: Any = None) -> np.ndarray:
    if threshold is None:
        try:
            result = binary_mask(image)
        except TypeError:
            result = binary_mask(image, None)
    else:
        try:
            result = binary_mask(image, threshold=threshold)
        except TypeError:
            result = binary_mask(image, threshold)
    return (np.asarray(result) > 0).astype(np.uint8) * 255


def _thin(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    thinning = getattr(getattr(cv2, "ximgproc", None), "thinning", None)
    if thinning is not None:
        return np.asarray(thinning(binary), dtype=np.uint8)
    try:
        from skimage.morphology import skeletonize

        return skeletonize(binary > 0).astype(np.uint8) * 255
    except ImportError:
        return binary


def _write_checked(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), _as_uint8(image))
    if not ok:
        raise IOError(f"failed to write collector output: {path}")


def _write_native_checked(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), np.asarray(image)):
        raise IOError(f"failed to write native-depth collector output: {path}")


def _resize_image(image: np.ndarray, resize: Any, interpolation: int) -> np.ndarray:
    if resize is None:
        return np.asarray(image)
    if isinstance(resize, Mapping):
        width = int(resize.get("width", image.shape[1]))
        height = int(resize.get("height", image.shape[0]))
    else:
        values = tuple(int(item) for item in resize)
        if len(values) != 2:
            raise ValueError("resize must contain width and height")
        width, height = values
    if width <= 0 or height <= 0:
        raise ValueError("resize dimensions must be positive")
    return np.asarray(cv2.resize(image, (width, height), interpolation=interpolation))


def _interpolation(value: Any) -> int:
    if isinstance(value, str):
        choices = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
        }
        key = value.casefold()
        if key not in choices:
            raise ValueError(f"unsupported resize interpolation: {value}")
        return choices[key]
    return int(value)


def _plot_profile(path_stem: Path, profile: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(
        profile["position_mm"],
        profile["foreground_rate"],
        color="#245a7a",
        linewidth=1.2,
    )
    axis.set_xlabel("Position (mm)")
    axis.set_ylabel("Detected collector foreground fraction")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path_stem.with_suffix(".png"), dpi=180)
    figure.savefig(path_stem.with_suffix(".svg"))
    plt.close(figure)


def _kernel_shape(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(int(item) for item in value)
        if len(values) == 2 and all(item > 0 for item in values):
            return values
    raise ValueError("kernel size must be a pair of positive integers")


def _intersection_mask(skeleton: np.ndarray) -> np.ndarray:
    binary = (np.asarray(skeleton) > 0).astype(np.uint8)
    neighbours = (
        cv2.filter2D(binary, cv2.CV_16S, np.ones((3, 3), dtype=np.int16)) - binary
    )
    # A skeleton crossing has at least three neighbouring pixels.  The
    # centre is retained in the result so each detected point is visible.
    return ((binary > 0) & (neighbours >= 3)).astype(np.uint8) * 255


def _intersection_nodes(mask: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Collapse connected junction pixels into graph nodes.

    ``intersection_pixels`` describes the raster output, whereas
    ``intersection_node_count`` describes connected junction regions.  The
    latter avoids counting every pixel in a thick or continuous crossing as a
    separate intersection.
    """

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (np.asarray(mask) > 0).astype(np.uint8), connectivity=8
    )
    rows: list[dict[str, Any]] = []
    for label in range(1, count):
        rows.append(
            {
                "node": label,
                "pixel_count": int(stats[label, cv2.CC_STAT_AREA]),
                "x": float(centroids[label, 0]),
                "y": float(centroids[label, 1]),
            }
        )
    return pd.DataFrame(rows, columns=["node", "pixel_count", "x", "y"]), max(
        0, count - 1
    )


def _component_summary(
    binary: np.ndarray, min_area: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = (np.asarray(binary) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    rows: list[dict[str, Any]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        rows.append(
            {
                "label": label,
                "area_px": area,
                "kept": bool(area >= min_area),
                "x": float(centroids[label, 0]),
                "y": float(centroids[label, 1]),
                "bbox_x": int(stats[label, cv2.CC_STAT_LEFT]),
                "bbox_y": int(stats[label, cv2.CC_STAT_TOP]),
                "bbox_width": int(stats[label, cv2.CC_STAT_WIDTH]),
                "bbox_height": int(stats[label, cv2.CC_STAT_HEIGHT]),
            }
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "label",
            "area_px",
            "kept",
            "x",
            "y",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
        ],
    )
    foreground = int(np.count_nonzero(mask))
    summary = {
        "connected_component_count": int(max(0, count - 1)),
        "kept_component_count": int(sum(item["kept"] for item in rows)),
        "foreground_pixels": foreground,
        "foreground_rate": float(foreground / mask.size) if mask.size else 0.0,
    }
    return table, summary


def _safe_ratio(numerator: float, denominator: float) -> float:
    try:
        value = safe_divide(numerator, denominator)
    except TypeError:
        value = safe_divide(numerator, denominator, default=0.0)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def run_collector_analysis(
    config: Any, general: Any, output_dir: str | Path
) -> dict[str, Any]:
    """Process a collector slice image and write all analysis artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source = _setting(config, general, "slices_path", None)
    if source is None:
        source = _setting(config, general, "input_path", None)
    if source is None:
        source = _setting(config, general, "collector_image", None)
    if source is None:
        raise ValueError("slices_path/input_path/collector_image is required")
    source_path = Path(source)
    if source_path.is_dir():
        source = source_path / "Slices.tif"
    image = np.asarray(read_grayscale(source))
    if image.ndim != 2:
        raise ValueError("read_grayscale must return a two-dimensional image")
    interpolation = _interpolation(
        _setting(config, general, "resize_interpolation", cv2.INTER_LINEAR)
    )
    resize = _setting(config, general, "resize", None)
    if resize is None:
        resize = _setting(config, general, "resize_size", None)
    resized = _resize_image(image, resize, interpolation)
    _write_native_checked(destination / "resized.tif", resized)

    threshold_method = _setting(config, general, "input_threshold_method", None)
    if threshold_method is None:
        threshold_method = (
            "fixed"
            if _setting(config, general, "threshold", None) is not None
            else "otsu"
        )
    threshold_method = str(threshold_method).casefold()
    if threshold_method == "otsu":
        input_threshold_used, binary_input = cv2.threshold(
            resized,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    elif threshold_method == "fixed":
        input_threshold = _setting(
            config,
            general,
            "input_threshold",
            _setting(config, general, "threshold", 0),
        )
        input_threshold_used, binary_input = cv2.threshold(
            resized,
            float(input_threshold),
            255,
            cv2.THRESH_BINARY,
        )
    else:
        raise ValueError("input_threshold_method must be 'otsu' or 'fixed'")
    binary_input = (np.asarray(binary_input) > 0).astype(np.uint8) * 255
    thinning_enabled = bool(_setting(config, general, "thinning", True))
    skeleton = _thin(binary_input) if thinning_enabled else binary_input
    _write_checked(destination / "skeleton.tif", skeleton)

    # The 4 x 3 kernel is parameterized but remains the historical default.
    kernel_values = _setting(config, general, "convolution_kernel", None)
    if kernel_values is None:
        kernel = np.tile(np.array([-1.0, 2.0, -1.0], dtype=np.float32), (4, 1))
    else:
        kernel = np.asarray(kernel_values, dtype=np.float32)
        if kernel.shape != (4, 3):
            raise ValueError("convolution_kernel must have shape (4, 3)")
    response_float = cv2.filter2D(skeleton.astype(np.float32), cv2.CV_32F, kernel)
    response_abs = np.abs(response_float)
    if response_abs.size and float(response_abs.max()) > float(response_abs.min()):
        response = cv2.normalize(response_abs, None, 0, 255, cv2.NORM_MINMAX)
    else:
        response = np.zeros(response_abs.shape, dtype=np.float32)
    response = _as_uint8(response)
    _write_checked(destination / "response.tif", response)

    response_threshold = _setting(config, general, "response_threshold", None)
    if response_threshold is None:
        threshold_used, binary = cv2.threshold(
            response,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    else:
        threshold_used, binary = cv2.threshold(
            response,
            float(response_threshold),
            255,
            cv2.THRESH_BINARY,
        )
    binary = np.asarray(binary, dtype=np.uint8)

    close_size = _kernel_shape(_setting(config, general, "close_kernel", None), (3, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_size)
    close_iterations = int(_setting(config, general, "close_iterations", 1))
    if close_iterations > 0:
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations
        )

    min_area = int(_setting(config, general, "min_area", 15))
    if min_area > 1:
        labels = np.zeros(binary.shape, dtype=np.int32)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (binary > 0).astype(np.uint8), connectivity=8
        )
        filtered = np.zeros(binary.shape, dtype=np.uint8)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                filtered[labels == label] = 255
        binary = filtered
    components, component_summary = _component_summary(binary, max(0, min_area))
    _write_checked(destination / "binary.tif", binary)
    _write_checked(destination / "binary.png", binary)
    intersections = _intersection_mask(skeleton)
    _write_checked(destination / "intersections.tif", intersections)
    _write_checked(destination / "intersections.png", intersections)
    intersection_table, intersection_node_count = _intersection_nodes(intersections)
    intersection_table.to_csv(destination / "intersection_nodes.csv", index=False)

    rows = np.arange(binary.shape[0], dtype=int)
    pixel_size_mm = float(_setting(config, general, "pixel_size_mm", 1.0))
    if not np.isfinite(pixel_size_mm) or pixel_size_mm <= 0:
        raise ValueError("pixel_size_mm must be finite and positive")
    foreground_count = np.count_nonzero(binary > 0, axis=1).astype(int)
    total_foreground = int(np.sum(foreground_count))
    profile = pd.DataFrame(
        {
            "row": rows,
            "position_mm": rows * pixel_size_mm,
            "foreground_pixels": foreground_count,
            "foreground_rate": (
                foreground_count / float(binary.shape[1]) if binary.shape[1] else 0.0
            ),
            "exposure_normalized": (
                foreground_count / float(total_foreground) if total_foreground else 0.0
            ),
        }
    )
    profile.to_csv(destination / "row_exposure_profile.csv", index=False)
    _plot_profile(destination / "row_exposure_profile", profile)
    components.to_csv(destination / "connected_components.csv", index=False)
    reference_comparison = None
    reference_source = _setting(config, general, "reference_path", None)
    if reference_source is not None:
        reference = np.asarray(read_grayscale(reference_source)) != 0
        if reference.shape != binary.shape:
            raise ValueError(
                f"collector reference shape {reference.shape} does not match output {binary.shape}"
            )
        current = binary > 0
        intersection_count = int(np.count_nonzero(current & reference))
        union_count = int(np.count_nonzero(current | reference))
        current_profile = np.mean(current, axis=1)
        reference_profile = np.mean(reference, axis=1)
        if np.std(current_profile) > 0 and np.std(reference_profile) > 0:
            pearson = float(np.corrcoef(current_profile, reference_profile)[0, 1])
            current_ranks = pd.Series(current_profile).rank(method="average").to_numpy()
            reference_ranks = (
                pd.Series(reference_profile).rank(method="average").to_numpy()
            )
            spearman = float(np.corrcoef(current_ranks, reference_ranks)[0, 1])
        else:
            pearson = spearman = None
        reference_comparison = {
            "reference_path": str(reference_source),
            "interpretation": "Regression diagnostic only; the historical processed mask is not treated as ground truth.",
            "current_foreground_pixels": int(np.count_nonzero(current)),
            "reference_foreground_pixels": int(np.count_nonzero(reference)),
            "pixel_jaccard": _safe_ratio(intersection_count, union_count),
            "current_pixels_covered_by_reference": _safe_ratio(
                intersection_count,
                int(np.count_nonzero(current)),
            ),
            "row_profile_pearson": pearson,
            "row_profile_spearman": spearman,
        }
    summary = {
        "input_path": str(source),
        "shape": [int(item) for item in binary.shape],
        "dtype_outputs": "uint8",
        "resize": resize,
        "source_dtype": str(image.dtype),
        "input_threshold_method": threshold_method,
        "input_threshold_used_source_units": float(input_threshold_used),
        "response_threshold": response_threshold,
        "response_threshold_used": float(threshold_used),
        "thinning": thinning_enabled,
        "convolution_kernel": kernel.tolist(),
        "close_kernel": list(close_size),
        "min_area": min_area,
        "connected_components": component_summary,
        "foreground_rate_after_filter": (
            float(np.count_nonzero(binary) / binary.size) if binary.size else 0.0
        ),
        "intersection_pixels": int(np.count_nonzero(intersections)),
        "intersection_node_count": int(intersection_node_count),
        "intersection_definition": "connected regions of skeleton pixels having at least three 8-neighbours",
        "historical_reference_comparison": reference_comparison,
        "outputs": {
            "resized": str(destination / "resized.tif"),
            "skeleton": str(destination / "skeleton.tif"),
            "response": str(destination / "response.tif"),
            "binary": str(destination / "binary.tif"),
            "binary_png": str(destination / "binary.png"),
            "intersections": str(destination / "intersections.tif"),
            "intersections_png": str(destination / "intersections.png"),
            "profile_csv": str(destination / "row_exposure_profile.csv"),
            "profile_png": str(destination / "row_exposure_profile.png"),
            "profile_svg": str(destination / "row_exposure_profile.svg"),
            "components_csv": str(destination / "connected_components.csv"),
            "intersection_nodes_csv": str(destination / "intersection_nodes.csv"),
            "summary_json": str(destination / "collector_summary.json"),
        },
    }
    _write_json(destination / "collector_summary.json", summary)
    return _json_safe(summary)
