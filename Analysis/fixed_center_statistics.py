"""Calculate curvature, exposure and region statistics from fixed-center mappings."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import normalization as normalization_module
import regions as region_module
import volume_sensitivity as resolution_module
from curvature_geometry import (
    archimedean_adjacent_turn_normal_separation,
    parallel_curve_curvature_interval,
)
from settings import (
    DATA_ROOT,
    OUTPUT_ROOT as ANALYSIS_OUTPUT_ROOT,
    relative_metadata,
)


FIT_INPUT_ROOT = DATA_ROOT / "curvature"
OBJECT_INPUT_ROOT = DATA_ROOT / "objects"
ROI_INPUT_ROOT = DATA_ROOT / "regions"
OUTPUT_ROOT = ANALYSIS_OUTPUT_ROOT / "fixed_center"
CURVATURE_ROOT = OUTPUT_ROOT / "curvature"
NORMALIZATION_ROOT = OUTPUT_ROOT / "normalization"
REGIONS_ROOT = OUTPUT_ROOT / "regions"
RESOLUTION_ROOT = OUTPUT_ROOT / "resolution"
PIXEL_SIZE_MM = 0.01684976915031592
VOXEL_VOLUME_MM3 = PIXEL_SIZE_MM**3
ELECTRODE_THICKNESS_PX = 29.0
ELECTRODE_THICKNESS_MM = ELECTRODE_THICKNESS_PX * PIXEL_SIZE_MM
AXIAL_SLICE_COUNT = 1591
AXIAL_SPAN_MM = AXIAL_SLICE_COUNT * PIXEL_SIZE_MM
LEGACY_CENTER_X_PIXEL = 613.0
LEGACY_CENTER_Y_PIXEL = 603.5
LEGACY_A_PIXEL = 427.3311730021125
LEGACY_B_PIXEL = 6.561336764513805
LEGACY_THETA_MIN = -48.84122951216121
LEGACY_THETA_MAX = 3.5256510425084278
DENSE_POINT_COUNT = 20_000
CLASS_ORDER = ["Internal", "Detach", "Surface"]
CLASS_FILES = {
    "Internal": "Internal_object_mapping.csv",
    "Detach": "Detach_object_mapping.csv",
    "Surface": "Surface_object_mapping.csv",
}

BIN_ORDER = ["< 0.2", "0.2 - 0.3", ">= 0.3"]
ANGLE_UPDATE_COLUMNS = {
    "arc_point_count",
    "arc_path_length_mm",
    "arc_path_share",
    "E_path",
    "path_proxy_volume_mm3",
    "defect_volume_fraction_path_proxy",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(relative_metadata(_jsonable(payload)), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _require_columns(
    table: pd.DataFrame,
    required: set[str],
    path: Path,
) -> None:
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _load_inputs() -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    summary_path = FIT_INPUT_ROOT / "fit_source_metadata.json"
    fit_points_path = FIT_INPUT_ROOT / "contour_fit_points.csv"
    ideal_path = FIT_INPUT_ROOT / "ideal_curve.csv"
    mapping_path = FIT_INPUT_ROOT / "contour_object_mapping.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not fit_points_path.is_file():
        raise FileNotFoundError(fit_points_path)
    if not ideal_path.is_file():
        raise FileNotFoundError(ideal_path)
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)

    comparison_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fit_points = pd.read_csv(fit_points_path)
    ideal_curve = pd.read_csv(ideal_path)
    mapped_legacy = pd.read_csv(mapping_path)
    _require_columns(
        fit_points,
        {
            "point_index_contour_order",
            "x",
            "y",
            "theta_unwrapped",
            "r_pixel",
            "r_fit_pixel",
            "radial_residual_pixel",
            "curvature_pixel",
            "curvature_1_per_mm",
        },
        fit_points_path,
    )
    _require_columns(
        ideal_curve,
        {"theta", "r_pixel", "x", "y", "curvature_pixel"},
        ideal_path,
    )
    _require_columns(
        mapped_legacy,
        {
            "feature_class",
            "object_index",
            "legacy_nearest_contour_index",
            "legacy_mapped_x",
            "legacy_mapped_y",
            "legacy_distance_px",
            "legacy_distance_mm",
            "legacy_r_pixel",
            "legacy_r_mm",
            "legacy_curvature_1_per_mm",
            "legacy_curvature_bin",
        },
        mapping_path,
    )

    current_summary_path = OBJECT_INPUT_ROOT / "source_geometry.json"
    current_summary = json.loads(current_summary_path.read_text(encoding="utf-8"))
    current_mappings: dict[str, pd.DataFrame] = {}
    legacy_mappings: dict[str, pd.DataFrame] = {}
    for class_name, filename in CLASS_FILES.items():
        current_path = OBJECT_INPUT_ROOT / filename
        current_table = pd.read_csv(current_path)
        legacy_table = mapped_legacy[
            mapped_legacy["feature_class"].astype(str) == class_name
        ].copy()
        if len(current_table) != len(legacy_table):
            raise ValueError(
                f"Object count mismatch for {class_name}: "
                f"{len(current_table)} versus {len(legacy_table)}"
            )
        current_mappings[class_name] = current_table
        legacy_mappings[class_name] = legacy_table

    return (
        comparison_summary,
        fit_points,
        ideal_curve,
        current_mappings,
        legacy_mappings,
        current_summary,
    )


def _classify_curvature(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    labels = np.full(array.shape, "invalid", dtype=object)
    finite = np.isfinite(array)
    labels[finite & (array < 0.2)] = BIN_ORDER[0]
    labels[finite & (array >= 0.2) & (array < 0.3)] = BIN_ORDER[1]
    labels[finite & (array >= 0.3)] = BIN_ORDER[2]
    return labels


def _distribution_stats(values: np.ndarray, scale: float = 1.0) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "rmse": None,
            "mae": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite) * scale),
        "median": float(np.median(finite) * scale),
        "mean": float(np.mean(finite) * scale),
        "rmse": float(np.sqrt(np.mean(finite**2)) * scale),
        "mae": float(np.mean(np.abs(finite)) * scale),
        "p95": float(np.percentile(finite, 95) * scale),
        "maximum": float(np.max(finite) * scale),
    }


def _signed_fit_metrics(
    residuals_pixel: np.ndarray,
    radius_pixel: np.ndarray,
) -> dict[str, float | None]:
    residuals = np.asarray(residuals_pixel, dtype=float)
    radius = np.asarray(radius_pixel, dtype=float)
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((radius - np.mean(radius)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else None
    return {
        "r2": r2,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "p95": float(np.percentile(np.abs(residuals), 95)),
        "max": float(np.max(np.abs(residuals))),
    }


def _build_legacy_fit(fit_points: pd.DataFrame) -> dict[str, Any]:
    points = fit_points.sort_values("point_index_contour_order")
    theta = points["theta_unwrapped"].to_numpy(dtype=float)
    radius = points["r_pixel"].to_numpy(dtype=float)
    residuals = points["radial_residual_pixel"].to_numpy(dtype=float)
    curvature_pixel = points["curvature_pixel"].to_numpy(dtype=float)
    metrics_pixel = _signed_fit_metrics(residuals, radius)
    metrics_mm = {
        key: (None if value is None else float(value) * PIXEL_SIZE_MM)
        for key, value in metrics_pixel.items()
    }
    metrics_mm["r2"] = metrics_pixel["r2"]
    theta_span = float(np.max(theta) - np.min(theta))
    radial_pitch_pixel = abs(2.0 * math.pi * LEGACY_B_PIXEL)
    covariance = np.asarray(
        json.loads(
            (FIT_INPUT_ROOT / "fit_source_metadata.json").read_text(encoding="utf-8")
        ).get("legacy_parameter_covariance_pixel", [[None, None], [None, None]]),
        dtype=object,
    )
    parameter_std_pixel: list[float | None] = []
    if covariance.shape == (2, 2):
        for index in range(2):
            value = covariance[index, index]
            parameter_std_pixel.append(
                float(math.sqrt(float(value))) if value is not None else None
            )
    return {
        "points": points,
        "theta": theta,
        "radius_pixel": radius,
        "residuals_pixel": residuals,
        "curvature_pixel": curvature_pixel,
        "curvature_mm": curvature_pixel / PIXEL_SIZE_MM,
        "fitted_radius_pixel": points["r_fit_pixel"].to_numpy(dtype=float),
        "center_xy": [LEGACY_CENTER_X_PIXEL, LEGACY_CENTER_Y_PIXEL],
        "a_pixel": LEGACY_A_PIXEL,
        "b_pixel": LEGACY_B_PIXEL,
        "a_mm": LEGACY_A_PIXEL * PIXEL_SIZE_MM,
        "b_mm": LEGACY_B_PIXEL * PIXEL_SIZE_MM,
        "theta_min": float(np.min(theta)),
        "theta_max": float(np.max(theta)),
        "theta_span": theta_span,
        "turn_count": theta_span / (2.0 * math.pi),
        "theta_direction": int(np.sign(theta[-1] - theta[0])),
        "radial_pitch_pixel": radial_pitch_pixel,
        "radial_pitch_mm": radial_pitch_pixel * PIXEL_SIZE_MM,
        "parameter_std_pixel": parameter_std_pixel,
        "parameter_std_mm": [
            None if value is None else value * PIXEL_SIZE_MM
            for value in parameter_std_pixel
        ],
        "metrics_pixel": metrics_pixel,
        "metrics_mm": metrics_mm,
        "parameter_covariance_pixel": covariance.tolist(),
    }


def _build_legacy_exposure(ideal_curve: pd.DataFrame) -> pd.DataFrame:
    curve = ideal_curve.copy().reset_index(drop=True)
    theta = curve["theta"].to_numpy(dtype=float)
    x = curve["x"].to_numpy(dtype=float)
    y = curve["y"].to_numpy(dtype=float)
    r_pixel = curve["r_pixel"].to_numpy(dtype=float)
    curvature_pixel = curve["curvature_pixel"].to_numpy(dtype=float)
    segment_length_pixel = np.zeros(len(curve), dtype=float)
    if len(curve) > 1:
        segment_length_pixel[1:] = np.hypot(np.diff(x), np.diff(y))
    segment_mid_curvature_pixel = curvature_pixel.copy()
    if len(curve) > 1:
        segment_mid_curvature_pixel[1:] = (
            curvature_pixel[1:] + curvature_pixel[:-1]
        ) / 2.0
    curvature_mm = curvature_pixel / PIXEL_SIZE_MM
    segment_mid_curvature_mm = segment_mid_curvature_pixel / PIXEL_SIZE_MM
    labels = _classify_curvature(segment_mid_curvature_mm)
    adjacent_separation_pixel = archimedean_adjacent_turn_normal_separation(
        theta,
        LEGACY_A_PIXEL,
        LEGACY_B_PIXEL,
    )
    return pd.DataFrame(
        {
            "theta": theta,
            "x": x,
            "y": y,
            "r_pixel": r_pixel,
            "r_mm": r_pixel * PIXEL_SIZE_MM,
            "curvature_pixel": curvature_pixel,
            "curvature_mm": curvature_mm,
            "adjacent_turn_normal_separation_px": adjacent_separation_pixel,
            "adjacent_turn_normal_separation_mm": (
                adjacent_separation_pixel * PIXEL_SIZE_MM
            ),
            "segment_length_pixel": segment_length_pixel,
            "segment_length_mm": segment_length_pixel * PIXEL_SIZE_MM,
            "curvature_bin": labels,
        }
    )


def _build_exposure_summary(exposure: pd.DataFrame) -> dict[str, Any]:
    path_by_bin = (
        exposure.groupby("curvature_bin", sort=False)["segment_length_mm"]
        .sum()
        .reindex(BIN_ORDER)
    )
    if path_by_bin.isna().any() or (path_by_bin <= 0.0).any():
        raise ValueError("Legacy exposure has a missing or non-positive curvature bin")
    total = float(path_by_bin.sum())
    return {
        "total_px": float(exposure["segment_length_pixel"].sum()),
        "total_mm": total,
        "by_bin": [
            {
                "curvature_bin": label,
                "path_length_px": float(
                    exposure.loc[
                        exposure["curvature_bin"] == label,
                        "segment_length_pixel",
                    ].sum()
                ),
                "path_length_mm": float(path_by_bin.loc[label]),
            }
            for label in BIN_ORDER
        ],
    }


def _build_mapping_table(
    class_name: str,
    current_table: pd.DataFrame,
    legacy_table: pd.DataFrame,
) -> pd.DataFrame:
    current = current_table.copy()
    legacy = legacy_table.sort_values("object_index").reset_index(drop=True)
    current = current.reset_index(drop=True)
    _require_columns(
        current,
        {
            "feature_class",
            "object_index",
            "Volume3d",
            "volume_voxel",
            "volume_mm3",
            "BaryCenterX",
            "BaryCenterY",
            "BaryCenterZ",
        },
        OBJECT_INPUT_ROOT / CLASS_FILES[class_name],
    )
    if not np.array_equal(
        current["object_index"].to_numpy(),
        legacy["object_index"].to_numpy(),
    ):
        raise ValueError(f"Object ordering changed for {class_name}")

    kappa = legacy["legacy_curvature_1_per_mm"].to_numpy(dtype=float)
    distances_px = legacy["legacy_distance_px"].to_numpy(dtype=float)
    parallel_low, parallel_high, interval_valid = parallel_curve_curvature_interval(
        kappa, ELECTRODE_THICKNESS_MM
    )
    center_bins = _classify_curvature(kappa)
    low_bins = _classify_curvature(parallel_low)
    high_bins = _classify_curvature(parallel_high)
    low_bins[~interval_valid] = "invalid"
    high_bins[~interval_valid] = "invalid"
    stable = (
        np.asarray(interval_valid, dtype=bool)
        & (low_bins == center_bins)
        & (high_bins == center_bins)
    )
    half_thickness_px = ELECTRODE_THICKNESS_PX / 2.0
    within_half_thickness = np.isfinite(distances_px) & (
        distances_px <= half_thickness_px
    )
    both = within_half_thickness & stable

    current["mapped_x"] = legacy["legacy_mapped_x"].to_numpy(dtype=float)
    current["mapped_y"] = legacy["legacy_mapped_y"].to_numpy(dtype=float)
    current["distance_px"] = distances_px
    current["distance_mm"] = legacy["legacy_distance_mm"].to_numpy(dtype=float)
    current["r_pixel"] = legacy["legacy_r_pixel"].to_numpy(dtype=float)
    current["r_mm"] = legacy["legacy_r_mm"].to_numpy(dtype=float)
    current["curvature_1_per_mm"] = kappa
    current["curvature_mm"] = kappa
    current["nominal_electrode_thickness_px"] = ELECTRODE_THICKNESS_PX
    current["nominal_electrode_thickness_mm"] = ELECTRODE_THICKNESS_MM
    current["curvature_parallel_low_1_per_mm"] = parallel_low
    current["curvature_parallel_high_1_per_mm"] = parallel_high
    current["curvature_parallel_interval_valid"] = interval_valid
    current["curvature_parallel_bin_low"] = low_bins
    current["curvature_parallel_bin_high"] = high_bins
    current["curvature_bin_stable_within_thickness"] = stable
    current["distance_within_half_thickness"] = within_half_thickness
    current["both_thickness_conditions"] = both
    current["curvature_bin"] = center_bins

    expected_volume_mm3 = (
        current["volume_voxel"].to_numpy(dtype=float) * VOXEL_VOLUME_MM3
    )
    if not np.allclose(
        current["volume_mm3"].to_numpy(dtype=float),
        expected_volume_mm3,
        rtol=1e-6,
        atol=1e-12,
    ):
        raise ValueError(f"volume_mm3 conversion mismatch for {class_name}")
    return current


def _build_curvature_bins(
    table: pd.DataFrame,
    exposure: pd.DataFrame,
) -> pd.DataFrame:
    exposure_by_bin = (
        exposure.groupby("curvature_bin", sort=False)["segment_length_mm"]
        .sum()
        .reindex(BIN_ORDER)
    )
    total_volume = float(table["volume_mm3"].sum())
    total_path = float(exposure_by_bin.sum())
    rows: list[dict[str, Any]] = []
    for label in BIN_ORDER:
        mask = table["curvature_bin"].astype(str).to_numpy() == label
        volume = float(table.loc[mask, "volume_mm3"].sum())
        count = int(mask.sum())
        path = float(exposure_by_bin.loc[label])
        path_fraction = path / total_path
        raw_proportion = volume / total_volume
        enrichment = raw_proportion / path_fraction
        rows.append(
            {
                "curvature_bin": label,
                "object_count": count,
                "defect_volume_mm3": volume,
                "available_path_length_px": float(
                    exposure.loc[
                        exposure["curvature_bin"] == label,
                        "segment_length_pixel",
                    ].sum()
                ),
                "available_path_length_mm": path,
                "path_length_fraction": path_fraction,
                "path_length_normalized_volume_mm3_per_mm": volume / path,
                "global_path_length_normalized_volume_mm3_per_mm": (
                    total_volume / total_path
                ),
                "raw_proportion": raw_proportion,
                "expected_volume_mm3_under_uniform_path_density": (
                    total_volume * path_fraction
                ),
                "observed_to_expected_ratio": enrichment,
                "log2_observed_to_expected": float(np.log2(enrichment)),
            }
        )
    return pd.DataFrame(rows)


def _thickness_summary_row(table: pd.DataFrame) -> dict[str, Any]:
    volumes = table["volume_mm3"].to_numpy(dtype=float)
    total_volume = float(volumes.sum())
    within = table["distance_within_half_thickness"].to_numpy(dtype=bool)
    stable = table["curvature_bin_stable_within_thickness"].to_numpy(dtype=bool)
    both = table["both_thickness_conditions"].to_numpy(dtype=bool)
    return {
        "feature_class": str(table["feature_class"].iloc[0]),
        "object_count": int(len(table)),
        "nominal_electrode_thickness_px": ELECTRODE_THICKNESS_PX,
        "nominal_electrode_thickness_mm": ELECTRODE_THICKNESS_MM,
        "distance_within_half_thickness_count": int(within.sum()),
        "distance_within_half_thickness_fraction": float(within.mean()),
        "curvature_bin_stable_within_thickness_count": int(stable.sum()),
        "curvature_bin_stable_within_thickness_fraction": float(stable.mean()),
        "both_conditions_count": int(both.sum()),
        "both_conditions_fraction": float(both.mean()),
        "volume_weighted_distance_within_half_thickness_fraction": float(
            volumes[within].sum() / total_volume
        ),
        "volume_weighted_curvature_bin_stable_within_thickness_fraction": float(
            volumes[stable].sum() / total_volume
        ),
        "volume_weighted_both_conditions_fraction": float(
            volumes[both].sum() / total_volume
        ),
    }


def _build_class_summary(
    class_name: str,
    table: pd.DataFrame,
    curvature_bins: pd.DataFrame,
    fit: dict[str, Any],
    exposure_summary: dict[str, Any],
) -> dict[str, Any]:
    distances = table["distance_px"].to_numpy(dtype=float)
    volumes = table["volume_mm3"].to_numpy(dtype=float)
    total_volume = float(volumes.sum())
    half_pitch = fit["radial_pitch_pixel"] / 2.0
    within = table["distance_within_half_thickness"].to_numpy(dtype=bool)
    stable = table["curvature_bin_stable_within_thickness"].to_numpy(dtype=bool)
    both = table["both_thickness_conditions"].to_numpy(dtype=bool)
    thickness = _thickness_summary_row(table)
    return {
        "object_count": int(len(table)),
        "input_volume_sum": float(table["volume_voxel"].sum()),
        "input_volume_unit": "voxel_count",
        "defect_volume_voxel": float(table["volume_voxel"].sum()),
        "defect_volume_mm3": total_volume,
        "mapping_distance_px": {
            "median": float(np.median(distances)),
            "p95": float(np.percentile(distances, 95)),
            "max": float(np.max(distances)),
        },
        "mapping_distance_mm": {
            "median": float(np.median(distances) * PIXEL_SIZE_MM),
            "p95": float(np.percentile(distances, 95) * PIXEL_SIZE_MM),
            "max": float(np.max(distances) * PIXEL_SIZE_MM),
        },
        "mapping_quality": {
            "half_pitch_threshold_px": half_pitch,
            "radial_pitch_px": fit["radial_pitch_pixel"],
            "distance_exceeds_half_pitch_count": int(
                np.count_nonzero(distances > half_pitch)
            ),
            "distance_exceeds_half_pitch_fraction": float(
                np.mean(distances > half_pitch)
            ),
            "interpretation": (
                "Distance is to the nearest extracted contour point; objects "
                "farther than half the fixed-fit radial pitch are flagged but retained."
            ),
        },
        "thickness_validation": {
            "nominal_electrode_thickness_px": ELECTRODE_THICKNESS_PX,
            "nominal_electrode_thickness_mm": ELECTRODE_THICKNESS_MM,
            "half_thickness_px": ELECTRODE_THICKNESS_PX / 2.0,
            "half_thickness_mm": ELECTRODE_THICKNESS_MM / 2.0,
            "distance_within_half_thickness_count": int(within.sum()),
            "distance_within_half_thickness_fraction": float(within.mean()),
            "curvature_bin_stable_within_thickness_count": int(stable.sum()),
            "curvature_bin_stable_within_thickness_fraction": float(stable.mean()),
            "both_conditions_count": int(both.sum()),
            "both_conditions_fraction": float(both.mean()),
            "parallel_curvature_interval_invalid_count": int(
                (~table["curvature_parallel_interval_valid"].to_numpy(dtype=bool)).sum()
            ),
            "volume_weighted": {
                "distance_within_half_thickness_fraction": float(
                    volumes[within].sum() / total_volume
                ),
                "curvature_bin_stable_within_thickness_fraction": float(
                    volumes[stable].sum() / total_volume
                ),
                "both_conditions_fraction": float(volumes[both].sum() / total_volume),
            },
            "interpretation": (
                "Flags describe centerline-mapping sensitivity to a nominal "
                "half-thickness offset."
            ),
        },
        "by_curvature_bin": curvature_bins.to_dict(orient="records"),
        "mapping_reference": "extracted_contour",
        "fit_residual_reference": "ideal_spiral",
        "thickness_summary": thickness,
        "exposure_reference": exposure_summary,
    }


def _build_curvature_plots(
    fit: dict[str, Any],
    fit_points: pd.DataFrame,
    ideal_curve: pd.DataFrame,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    points = fit_points.sort_values("point_index_contour_order")
    overlay, axis = plt.subplots(figsize=(8.0, 7.0), dpi=160)
    axis.plot(
        points["x"],
        points["y"],
        color="black",
        linewidth=0.35,
        alpha=0.65,
        label="extracted contour",
    )
    axis.plot(
        ideal_curve["x"],
        ideal_curve["y"],
        color="#D55E00",
        linewidth=0.9,
        label="fixed-center ideal spiral",
    )
    axis.scatter(
        [LEGACY_CENTER_X_PIXEL],
        [LEGACY_CENTER_Y_PIXEL],
        marker="+",
        color="#0072B2",
        s=55,
        label="fixed centre (613, 603.5)",
    )
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_xlabel("x (processed image pixel)")
    axis.set_ylabel("y (processed image pixel; origin upper)")
    axis.set_title("Fixed-center spiral for curvature statistics")
    axis.legend(frameon=False, fontsize=8)
    overlay.tight_layout()
    path = CURVATURE_ROOT / "fit_overlay.png"
    overlay.savefig(path, dpi=220, bbox_inches="tight")
    paths["fit_overlay_png"] = str(path)
    plt.close(overlay)

    residual, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    theta = points["theta_unwrapped"].to_numpy(dtype=float)
    residual_mm = points["radial_residual_mm"].to_numpy(dtype=float)
    axes[0].plot(theta, residual_mm, color="#0072B2", linewidth=0.45)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_xlabel("raw unwrapped theta (rad; contour order)")
    axes[0].set_ylabel("radial residual (mm)")
    axes[0].set_title("Native radial residual")
    axes[1].hist(np.abs(residual_mm), bins=45, color="#D55E00", alpha=0.8)
    axes[1].set_xlabel("absolute radial residual (mm)")
    axes[1].set_ylabel("contour-point count")
    axes[1].set_title("Residual magnitude")
    residual.tight_layout()
    path = CURVATURE_ROOT / "radial_residual.png"
    residual.savefig(path, dpi=220, bbox_inches="tight")
    paths["radial_residual_png"] = str(path)
    plt.close(residual)
    return paths


def _write_curvature_outputs(
    fit: dict[str, Any],
    fit_points: pd.DataFrame,
    ideal_curve: pd.DataFrame,
    current_mappings: dict[str, pd.DataFrame],
    legacy_mappings: dict[str, pd.DataFrame],
    current_summary: dict[str, Any],
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, Any],
    Path,
]:
    CURVATURE_ROOT.mkdir(parents=True, exist_ok=True)
    exposure = _build_legacy_exposure(ideal_curve)
    exposure_summary = _build_exposure_summary(exposure)
    exposure_path = CURVATURE_ROOT / "fitted_curve_exposure.csv"
    exposure.to_csv(exposure_path, index=False, float_format="%.15g")

    output_mappings: dict[str, pd.DataFrame] = {}
    class_summaries: dict[str, Any] = {}
    thickness_rows: list[dict[str, Any]] = []
    bin_paths: dict[str, str] = {}
    for class_name in CLASS_ORDER:
        table = _build_mapping_table(
            class_name,
            current_mappings[class_name],
            legacy_mappings[class_name],
        )
        output_mappings[class_name] = table
        mapping_path = CURVATURE_ROOT / CLASS_FILES[class_name]
        table.to_csv(mapping_path, index=False, float_format="%.15g")
        curvature_bins = _build_curvature_bins(table, exposure)
        bins_path = CURVATURE_ROOT / f"{class_name}_curvature_bins.csv"
        curvature_bins.to_csv(bins_path, index=False, float_format="%.15g")
        bin_paths[class_name] = str(bins_path)
        class_summaries[class_name] = _build_class_summary(
            class_name,
            table,
            curvature_bins,
            fit,
            exposure_summary,
        )
        class_summaries[class_name]["mapping_csv"] = str(mapping_path)
        class_summaries[class_name]["curvature_bins_csv"] = str(bins_path)
        thickness_rows.append(_thickness_summary_row(table))

    thickness_path = CURVATURE_ROOT / "thickness_mapping_summary.csv"
    pd.DataFrame(thickness_rows).to_csv(
        thickness_path,
        index=False,
        float_format="%.15g",
    )
    plot_paths = _build_curvature_plots(fit, fit_points, ideal_curve)

    residuals = fit["residuals_pixel"]
    pitch_pixel = fit["radial_pitch_pixel"]
    metrics_pixel = fit["metrics_pixel"]
    rmse_fraction = metrics_pixel["rmse"] / pitch_pixel
    p95_fraction = metrics_pixel["p95"] / pitch_pixel
    separation_px = exposure["adjacent_turn_normal_separation_px"].to_numpy(dtype=float)
    spacing_px = exposure["segment_length_pixel"].to_numpy(dtype=float)
    fit_diagnostics = {
        "r2": metrics_pixel["r2"],
        "rmse_pitch_fraction": rmse_fraction,
        "p95_pitch_fraction": p95_fraction,
    }
    summary = {
        "analysis": "fixed_center_statistics",
        "source": "Fixed-center contour and object mapping tables",
        "method": {
            "center_xy": [LEGACY_CENTER_X_PIXEL, LEGACY_CENTER_Y_PIXEL],
            "a_pixel": LEGACY_A_PIXEL,
            "b_pixel": LEGACY_B_PIXEL,
            "theta_min": LEGACY_THETA_MIN,
            "theta_max": LEGACY_THETA_MAX,
            "theta_formula": "atan2(cy-y, x-cx), then np.unwrap in contour order",
            "theta_direction": int(fit["theta_direction"]),
            "theta_direction_semantics": (
                "Sign of angular progression along the contour traversal order"
            ),
            "ideal_curve_coordinates": "x=cx+r*cos(theta), y=cy-r*sin(theta)",
            "mapping_reference": "extracted_contour",
            "fit_residual_reference": "ideal_spiral",
            "radial_pitch_definition": "abs(2*pi*b_pixel)",
            "half_pitch_flag_threshold": "abs(2*pi*b_pixel)/2",
        },
        "inputs": {
            "fit_metadata": str(FIT_INPUT_ROOT / "fit_source_metadata.json"),
            "contour_points": str(FIT_INPUT_ROOT / "contour_fit_points.csv"),
            "ideal_curve": str(FIT_INPUT_ROOT / "ideal_curve.csv"),
            "object_mapping": str(FIT_INPUT_ROOT / "contour_object_mapping.csv"),
        },
        "fit_diagnostics": fit_diagnostics,
        "transform": current_summary.get("transform"),
        "center_xy": [LEGACY_CENTER_X_PIXEL, LEGACY_CENTER_Y_PIXEL],
        "center_initial_xy": [LEGACY_CENTER_X_PIXEL, LEGACY_CENTER_Y_PIXEL],
        "center_shift_pixel": 0.0,
        "pixel_size_mm": PIXEL_SIZE_MM,
        "voxel_volume_mm3": VOXEL_VOLUME_MM3,
        "input_volume_unit": "voxel_count",
        "thinning": "saved extracted contour consumed",
        "fit": {
            "a_pixel": LEGACY_A_PIXEL,
            "b_pixel": LEGACY_B_PIXEL,
            "a_mm": fit["a_mm"],
            "b_mm": fit["b_mm"],
            "theta_min": fit["theta_min"],
            "theta_max": fit["theta_max"],
            "theta_span": fit["theta_span"],
            "theta_start_is_nonzero": bool(abs(fit["theta_min"]) > 1.0e-12),
            "dense_theta_range_source": (
                "legacy recorded theta bounds; 20,000 ideal samples"
            ),
            "theta_direction": fit["theta_direction"],
            "radial_pitch_initial_pixel": fit["radial_pitch_pixel"],
            "radial_pitch_pixel": fit["radial_pitch_pixel"],
            "parameter_std_pixel": fit["parameter_std_pixel"],
            "parameter_std_mm": fit["parameter_std_mm"],
            "parameter_uncertainty_note": (
                "Square roots of the recorded parameter-covariance diagonal"
            ),
            "point_count": int(len(fit_points)),
            "metrics_pixel": fit["metrics_pixel"],
            "metrics_mm": fit["metrics_mm"],
        },
        "curvature_formula": (
            "kappa_pixel=(r_pixel^2+2*b_pixel^2)/(r_pixel^2+b_pixel^2)^(3/2); "
            "kappa_mm=kappa_pixel/pixel_size_mm"
        ),
        "dense_curve_point_spacing": {
            "segment_count": int(max(0, len(exposure) - 1)),
            "minimum_px": float(np.min(spacing_px[1:])),
            "median_px": float(np.median(spacing_px[1:])),
            "p95_px": float(np.percentile(spacing_px[1:], 95)),
            "maximum_px": float(np.max(spacing_px[1:])),
            "minimum_mm": float(np.min(spacing_px[1:]) * PIXEL_SIZE_MM),
            "median_mm": float(np.median(spacing_px[1:]) * PIXEL_SIZE_MM),
            "p95_mm": float(np.percentile(spacing_px[1:], 95) * PIXEL_SIZE_MM),
            "maximum_mm": float(np.max(spacing_px[1:]) * PIXEL_SIZE_MM),
            "interpretation": "spacing of sampled ideal-spiral points",
        },
        "adjacent_turn_normal_separation": {
            "minimum_px": float(np.min(separation_px)),
            "median_px": float(np.median(separation_px)),
            "p95_px": float(np.percentile(separation_px, 95)),
            "maximum_px": float(np.max(separation_px)),
            "minimum_mm": float(np.min(separation_px) * PIXEL_SIZE_MM),
            "median_mm": float(np.median(separation_px) * PIXEL_SIZE_MM),
            "p95_mm": float(np.percentile(separation_px, 95) * PIXEL_SIZE_MM),
            "maximum_mm": float(np.max(separation_px) * PIXEL_SIZE_MM),
            "normal_projection_method": (
                "same-angle radial pitch projected onto the local Archimedean normal"
            ),
            "nominal_electrode_thickness_px": ELECTRODE_THICKNESS_PX,
            "nominal_electrode_thickness_mm": ELECTRODE_THICKNESS_MM,
            "nominal_thickness_bands_overlap": bool(
                np.min(separation_px) < ELECTRODE_THICKNESS_PX
            ),
            "overlap_criterion": (
                "minimum estimated normal separation < nominal total thickness"
            ),
        },
        "curvature_bin_edges": [None, 0.2, 0.3, None],
        "available_path_length": exposure_summary,
        "classes": class_summaries,
        "plots": plot_paths,
        "outputs": {
            "curve_exposure_csv": str(exposure_path),
            "thickness_mapping_summary_csv": str(thickness_path),
            "mapping_csv": {
                class_name: str(CURVATURE_ROOT / filename)
                for class_name, filename in CLASS_FILES.items()
            },
            "curvature_bins_csv": bin_paths,
        },
    }
    summary_path = CURVATURE_ROOT / "curvature_summary.json"
    summary["outputs"]["summary_json"] = str(summary_path)
    _write_json(summary_path, summary)
    return output_mappings, exposure, summary, summary_path


def _write_normalization_outputs(
    mappings: dict[str, pd.DataFrame],
    exposure: pd.DataFrame,
) -> tuple[pd.DataFrame, Path]:
    NORMALIZATION_ROOT.mkdir(parents=True, exist_ok=True)
    exposure_table = (
        exposure.groupby("curvature_bin", sort=False)["segment_length_mm"]
        .sum()
        .reindex(BIN_ORDER)
        .rename("available_path_length_mm")
        .reset_index()
    )
    normalization = normalization_module._build_normalization_table(
        mappings,
        exposure_table,
    )
    normalization_path = NORMALIZATION_ROOT / "defect_exposure_normalization.csv"
    normalization.to_csv(normalization_path, index=False, float_format="%.15g")
    normalization_module._make_plot(normalization, NORMALIZATION_ROOT)
    exposure_table.to_csv(
        NORMALIZATION_ROOT / "fitted_arc_exposure.csv",
        index=False,
        float_format="%.15g",
    )
    _write_json(
        NORMALIZATION_ROOT / "metadata.json",
        {
            "analysis": "legacy_fixed_center_normalization",
            "mapping_reference": "fixed-center nearest extracted contour point",
            "fit_residual_reference": "fixed-center ideal spiral",
            "normalization_function": (
                "revision_normalization_20260906._build_normalization_table"
            ),
            "inputs": {
                "curvature_root": str(CURVATURE_ROOT),
                "mapping_files": {
                    class_name: str(CURVATURE_ROOT / filename)
                    for class_name, filename in CLASS_FILES.items()
                },
                "fitted_curve_exposure": str(
                    CURVATURE_ROOT / "fitted_curve_exposure.csv"
                ),
            },
            "exposure_denominator": "fixed-center fitted-curve exposure",
            "geometric_proxy_formula": (
                "available_path_length_mm * electrode_thickness_mm * axial_span_mm"
            ),
            "output_csv": str(normalization_path),
        },
    )
    return normalization, normalization_path


def _update_angle_region_cache(
    exposure: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]], dict[str, str]]:
    REGIONS_ROOT.mkdir(parents=True, exist_ok=True)
    angle_path = ROI_INPUT_ROOT / "angle_region_comparison.csv"
    old_angle = pd.read_csv(angle_path)
    old_angle_text = pd.read_csv(
        angle_path,
        dtype=str,
        keep_default_na=False,
    )
    _require_columns(
        old_angle,
        {
            "scope",
            "region",
            "n_slices",
            "annulus_volume_share",
            "positive_volume_mm3",
            "path_proxy_thickness_mm",
            "path_proxy_axial_span_mm",
        },
        angle_path,
    )
    arc_exposure = region_module._build_arc_angular_exposure(exposure)
    updated = old_angle.copy()
    for index, row in updated.iterrows():
        region_name = str(row["region"])
        if region_name not in {"sector", "complement"}:
            raise ValueError(f"Unexpected angle region {region_name}")
        arc = arc_exposure[region_name]
        path_share = float(arc["arc_path_share"])
        path_length = float(arc["arc_path_length_mm"])
        n_slices = int(row["n_slices"])
        proxy_volume = path_length * ELECTRODE_THICKNESS_MM * (n_slices * PIXEL_SIZE_MM)
        updated.loc[index, "arc_point_count"] = int(arc["arc_point_count"])
        updated.loc[index, "arc_path_length_mm"] = path_length
        updated.loc[index, "arc_path_share"] = path_share
        updated.loc[index, "E_path"] = float(row["annulus_volume_share"]) / path_share
        updated.loc[index, "path_proxy_volume_mm3"] = proxy_volume
        updated.loc[index, "defect_volume_fraction_path_proxy"] = (
            float(row["positive_volume_mm3"]) / proxy_volume
        )
    updated_path = REGIONS_ROOT / "angle_region_comparison.csv"
    updated_text = old_angle_text.copy()
    for index in updated.index:
        updated_text.loc[index, "arc_point_count"] = str(
            int(updated.loc[index, "arc_point_count"])
        )
        for column in [
            "arc_path_length_mm",
            "arc_path_share",
            "E_path",
            "path_proxy_volume_mm3",
            "defect_volume_fraction_path_proxy",
        ]:
            updated_text.loc[index, column] = (
                f"{float(updated.loc[index, column]):.15g}"
            )
    updated_text.to_csv(updated_path, index=False)

    copied_paths: dict[str, str] = {}
    for filename in [
        "angle_scope_summary.csv",
        "angle_5deg_histogram.csv",
        "angle_slice_counts.csv",
    ]:
        source = ROI_INPUT_ROOT / filename
        destination = REGIONS_ROOT / filename
        shutil.copyfile(source, destination)
        copied_paths[filename] = str(destination)
    return updated, arc_exposure, copied_paths


def _write_region_outputs(
    mappings: dict[str, pd.DataFrame],
    exposure: pd.DataFrame,
    normalization_path: Path,
) -> dict[str, Any]:
    REGIONS_ROOT.mkdir(parents=True, exist_ok=True)
    curve_parameters = region_module._derive_curve_parameters(exposure)
    resolved_schemes = region_module._resolve_radial_schemes(curve_parameters)
    region_module.OLD_NORMALIZATION_PATH = normalization_path
    global_normalization = region_module._build_global_normalization()
    radial_table = region_module._build_radial_table(
        mappings,
        exposure,
        resolved_schemes,
    )
    sensitivity_table = region_module._build_curvature_sensitivity(
        mappings,
        exposure,
    )
    updated_angle, arc_exposure, copied_paths = _update_angle_region_cache(exposure)

    radial_path = REGIONS_ROOT / "radial_region_metrics.csv"
    sensitivity_path = REGIONS_ROOT / "curvature_threshold_sensitivity.csv"
    parameters_path = REGIONS_ROOT / "curve_parameters.csv"
    global_path = REGIONS_ROOT / "global_exposure_normalization.csv"
    radial_table.to_csv(radial_path, index=False, float_format="%.15g")
    sensitivity_table.to_csv(
        sensitivity_path,
        index=False,
        float_format="%.15g",
    )
    global_normalization.to_csv(global_path, index=False, float_format="%.15g")
    region_module._write_curve_parameter_table(curve_parameters, parameters_path)

    first_turn = radial_table[
        (radial_table["scheme"] == "first_observed_turn")
        & (radial_table["feature_class"] == "Internal")
        & (radial_table["region"] == "inner")
    ].iloc[0]
    angle_epath = {
        str(scope): float(
            updated_angle.loc[
                (updated_angle["scope"] == scope)
                & (updated_angle["region"] == "sector"),
                "E_path",
            ].iloc[0]
        )
        for scope in region_module.ANGLE_SCOPES
    }
    summary = {
        "analysis": "fixed_center_region_scope",
        "method": {
            "mapping_reference": "fixed-center nearest extracted contour point",
            "fit_residual_reference": "fixed-center ideal spiral",
            "downstream_functions_reused": [
                "_derive_curve_parameters",
                "_resolve_radial_schemes",
                "_build_radial_table",
                "_build_curvature_sensitivity",
                "_build_arc_angular_exposure",
                "_build_global_normalization",
            ],
            "global_normalization_input": str(normalization_path),
        },
        "inputs": {
            "curvature_root": str(CURVATURE_ROOT),
            "old_angle_region_comparison": str(
                ROI_INPUT_ROOT / "angle_region_comparison.csv"
            ),
        },
        "curve_parameters": curve_parameters,
        "resolved_radial_schemes": {
            name: {"lower_mm": lower, "upper_mm_exclusive": upper}
            for name, (lower, upper) in resolved_schemes.items()
        },
        "first_observed_turn_internal": {
            key: _jsonable(value) for key, value in first_turn.to_dict().items()
        },
        "angle_arc_exposure": arc_exposure,
        "angle_scope_epath_sector": angle_epath,
        "outputs": {
            "radial_region_metrics": str(radial_path),
            "curvature_threshold_sensitivity": str(sensitivity_path),
            "curve_parameters": str(parameters_path),
            "global_exposure_normalization": str(global_path),
            "angle_region_comparison": str(
                REGIONS_ROOT / "angle_region_comparison.csv"
            ),
            **copied_paths,
        },
        "updated_angle_columns": sorted(ANGLE_UPDATE_COLUMNS),
    }
    summary_path = REGIONS_ROOT / "revision_region_scope_summary.json"
    _write_json(summary_path, summary)
    return summary


def _run_resolution() -> dict[str, Any]:
    RESOLUTION_ROOT.mkdir(parents=True, exist_ok=True)
    resolution_module.INPUT_ROOT = CURVATURE_ROOT
    resolution_module.OUTPUT_ROOT = RESOLUTION_ROOT
    resolution_module.main()
    metadata_path = RESOLUTION_ROOT / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _build_checks(
    current_mappings: dict[str, pd.DataFrame],
    mappings: dict[str, pd.DataFrame],
    exposure: pd.DataFrame,
    normalization: pd.DataFrame,
    angle_region: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for class_name in CLASS_ORDER:
        current = current_mappings[class_name].reset_index(drop=True)
        legacy = mappings[class_name].reset_index(drop=True)
        checks[f"{class_name}_object_identity_and_order_unchanged"] = bool(
            np.array_equal(
                current["object_index"].to_numpy(),
                legacy["object_index"].to_numpy(),
            )
            and np.allclose(
                current[
                    [
                        "Volume3d",
                        "volume_voxel",
                        "BaryCenterX",
                        "BaryCenterY",
                        "BaryCenterZ",
                    ]
                ].to_numpy(dtype=float),
                legacy[
                    [
                        "Volume3d",
                        "volume_voxel",
                        "BaryCenterX",
                        "BaryCenterY",
                        "BaryCenterZ",
                    ]
                ].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            )
        )
        checks[f"{class_name}_volume_mm3_unchanged"] = bool(
            np.allclose(
                current["volume_mm3"].to_numpy(dtype=float),
                legacy["volume_mm3"].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            )
        )
        checks[f"{class_name}_bin_count_conservation"] = bool(
            int(legacy["curvature_bin"].value_counts().sum()) == len(legacy)
        )
    segment_lengths = exposure["segment_length_mm"].to_numpy(dtype=float)
    checks["all_segment_lengths_nonnegative"] = bool(np.all(segment_lengths >= 0.0))
    path_by_bin = (
        exposure.groupby("curvature_bin", sort=False)["segment_length_mm"]
        .sum()
        .reindex(BIN_ORDER)
    )
    checks["all_curvature_bin_path_lengths_positive"] = bool(
        path_by_bin.notna().all() and (path_by_bin > 0.0).all()
    )
    checks["normalization_path_denominator_uses_new_exposure"] = bool(
        np.allclose(
            normalization.drop_duplicates("curvature_bin")
            .set_index("curvature_bin")
            .loc[BIN_ORDER, "available_path_length_mm"]
            .to_numpy(dtype=float),
            path_by_bin.loc[BIN_ORDER].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
    )
    checks["angle_scope_rows_preserved"] = bool(
        set(angle_region["scope"].astype(str)) == set(region_module.ANGLE_SCOPES)
        and set(angle_region["region"].astype(str)) == {"sector", "complement"}
    )
    checks["structural_checks_all_pass"] = all(
        bool(value)
        for key, value in checks.items()
        if key != "structural_checks_all_pass"
    )
    return checks


def run(output_root: Path | None = None) -> dict[str, Any]:
    """Calculate curvature mappings, exposure normalization and regional statistics."""
    global OUTPUT_ROOT, CURVATURE_ROOT, NORMALIZATION_ROOT, REGIONS_ROOT, RESOLUTION_ROOT
    if output_root is not None:
        OUTPUT_ROOT = Path(output_root)
        CURVATURE_ROOT = OUTPUT_ROOT / "curvature"
        NORMALIZATION_ROOT = OUTPUT_ROOT / "normalization"
        REGIONS_ROOT = OUTPUT_ROOT / "regions"
        RESOLUTION_ROOT = OUTPUT_ROOT / "resolution"
    (
        source_metadata,
        fit_points,
        ideal_curve,
        physical_mappings,
        contour_mappings,
        source_geometry,
    ) = _load_inputs()
    fit = _build_legacy_fit(fit_points)
    mappings, exposure, curvature_summary, _ = _write_curvature_outputs(
        fit,
        fit_points,
        ideal_curve,
        physical_mappings,
        contour_mappings,
        source_geometry,
    )
    normalization, normalization_path = _write_normalization_outputs(mappings, exposure)
    region_summary = _write_region_outputs(mappings, exposure, normalization_path)
    resolution_metadata = _run_resolution()
    checks = _build_checks(
        physical_mappings,
        mappings,
        exposure,
        normalization,
        pd.read_csv(REGIONS_ROOT / "angle_region_comparison.csv"),
    )
    summary = {
        "analysis": "fixed_center_statistics",
        "input_level": "Saved contour fit, segmented object identity/volume/centroid tables and recorded angular ROI voxel counts",
        "primary_curvature_method": "fixed-center Archimedean spiral; nearest extracted-contour object mapping",
        "classes": CLASS_ORDER,
        "center_pixels": [LEGACY_CENTER_X_PIXEL, LEGACY_CENTER_Y_PIXEL],
        "fit": curvature_summary["fit"],
        "units": {"length": "mm", "volume": "mm3", "curvature": "1/mm"},
        "outputs": {
            "curvature": str(CURVATURE_ROOT),
            "normalization": str(NORMALIZATION_ROOT),
            "regions": str(REGIONS_ROOT),
            "resolution": str(RESOLUTION_ROOT),
        },
        "checks": checks,
        "region_parameters": region_summary["curve_parameters"],
        "baseline_totals": resolution_metadata["baseline_totals"],
    }
    _write_json(OUTPUT_ROOT / "summary.json", summary)
    print("Fixed-center statistics:", OUTPUT_ROOT)
    return summary


if __name__ == "__main__":
    run()
