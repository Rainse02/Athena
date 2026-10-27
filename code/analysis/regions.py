"""Regional exposure and curvature-boundary sensitivity on the selected path."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from package_paths import OUTPUT_ROOT, relative_metadata


OLD_NORMALIZATION_PATH = (
    OUTPUT_ROOT / "fixed_center" / "normalization" / "defect_exposure_normalization.csv"
)

PIXEL_SIZE_MM = 0.01684976915031592

VOXEL_VOLUME_MM3 = PIXEL_SIZE_MM**3

ELECTRODE_THICKNESS_MM = 0.4886433017

AXIAL_SLICE_COUNT = 1591

ANGLE_CENTER_X_PIXEL = 613.0

ANGLE_CENTER_Y_PIXEL = 603.0

ANGLE_RADIAL_FLOOR_MIN = 201

ANGLE_RADIAL_FLOOR_MAX_EXCLUSIVE = 480

ANGLE_SECTOR_LOW_DEG = 80.0

ANGLE_SECTOR_HIGH_DEG = 140.0

ANGLE_BIN_WIDTH_DEG = 5.0

ANGLE_BIN_COUNT = 72

CLASS_FILES = {
    "Internal": "Internal_object_mapping.csv",
    "Detach": "Detach_object_mapping.csv",
    "Surface": "Surface_object_mapping.csv",
}

CLASS_ORDER = list(CLASS_FILES)

CURVATURE_BIN_ORDER = ["< low", "low - high", ">= high"]

ANGLE_SCOPES = {
    "f4f_510_1009": (510, 1009),
    "matched_control_1010_2100": (1010, 2100),
    "full_510_2100": (510, 2100),
}

THRESHOLD_LOWS = (0.18, 0.20, 0.22)

THRESHOLD_HIGHS = (0.27, 0.30, 0.33)

RADIAL_SCHEMES = {
    "first_observed_turn": (None, None),
    "exploratory_lower": (None, 2.3),
    "exploratory_upper": (None, 2.7),
}


def _require_columns(table: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _derive_curve_parameters(exposure: pd.DataFrame) -> dict[str, float | str]:
    first = exposure.iloc[0]
    last = exposure.iloc[-1]
    theta_first = float(first["theta"])
    theta_last = float(last["theta"])
    r_first = float(first["r_mm"])
    r_last = float(last["r_mm"])
    if not theta_last > theta_first:
        raise ValueError("Fitted curve theta endpoints are not increasing")
    b_mm = (r_last - r_first) / (theta_last - theta_first)
    a_mm = r_first - b_mm * theta_first
    pitch_mm = 2.0 * math.pi * b_mm
    first_turn_upper_mm = r_first + pitch_mm

    def curvature_at_radius(radius_mm: float) -> float:
        numerator = radius_mm**2 + 2.0 * b_mm**2
        denominator = (radius_mm**2 + b_mm**2) ** 1.5
        return numerator / denominator

    def radius_at_curvature(target: float) -> float:
        return brentq(
            lambda radius: curvature_at_radius(radius) - target,
            1.0e-12,
            100.0,
        )

    radius_at_020 = radius_at_curvature(0.20)
    radius_at_030 = radius_at_curvature(0.30)
    return {
        "fit_a_mm": float(a_mm),
        "fit_b_mm": float(b_mm),
        "r_min_mm": float(r_first),
        "r_last_mm": float(r_last),
        "theta_min_rad": float(theta_first),
        "theta_max_rad": float(theta_last),
        "pitch_2pi_b_mm": float(pitch_mm),
        "first_observed_turn_upper_mm": float(first_turn_upper_mm),
        "radius_at_kappa_0.2_mm": float(radius_at_020),
        "radius_at_kappa_0.3_mm": float(radius_at_030),
        "kappa_at_r_2_mm_inverse": float(curvature_at_radius(2.0)),
        "parameterization": "r_mm = fit_a_mm + fit_b_mm * theta_rad, using the first and last fitted exposure rows",
        "curvature_formula": "kappa_1_per_mm = (r^2 + 2*b^2) / (r^2 + b^2)^(3/2)",
    }


def _resolve_radial_schemes(
    curve_parameters: dict[str, float | str],
) -> dict[str, tuple[float, float]]:
    r_min = float(curve_parameters["r_min_mm"])
    first_turn_upper = float(curve_parameters["first_observed_turn_upper_mm"])
    resolved: dict[str, tuple[float, float]] = {}
    for name, (lower, upper) in RADIAL_SCHEMES.items():
        if lower is None:
            lower_value = r_min
        else:
            lower_value = float(lower)
        if upper is None:
            upper_value = first_turn_upper
        else:
            upper_value = float(upper)
        if not upper_value > lower_value:
            raise ValueError(
                f"Invalid radial scheme {name}: {lower_value}, {upper_value}"
            )
        resolved[name] = (lower_value, upper_value)
    return resolved


def _build_global_normalization() -> pd.DataFrame:
    if not OLD_NORMALIZATION_PATH.is_file():
        raise FileNotFoundError(OLD_NORMALIZATION_PATH)
    old = pd.read_csv(OLD_NORMALIZATION_PATH)
    _require_columns(
        old,
        {
            "feature_class",
            "curvature_bin",
            "object_count",
            "raw_volume_mm3",
            "available_path_length_mm",
            "raw_volume_share",
            "raw_count_share",
            "path_length_share",
            "volume_share_to_length_share",
            "count_share_to_length_share",
        },
        OLD_NORMALIZATION_PATH,
    )
    rows: list[dict[str, float | int | str]] = []
    for class_name in CLASS_ORDER:
        source = old[old["feature_class"] == class_name].copy()
        if source.empty:
            raise ValueError(f"No old normalization rows for {class_name}")
        source["volume_rate"] = (
            source["raw_volume_mm3"] / source["available_path_length_mm"]
        )
        source["count_rate"] = (
            source["object_count"] / source["available_path_length_mm"]
        )
        volume_rate_sum = float(source["volume_rate"].sum())
        count_rate_sum = float(source["count_rate"].sum())
        class_volume = float(source["raw_volume_mm3"].sum())
        class_count = int(source["object_count"].sum())
        class_mean_volume = class_volume / class_count
        for _, row in source.iterrows():
            object_count = int(row["object_count"])
            volume_mm3 = float(row["raw_volume_mm3"])
            mean_volume = volume_mm3 / object_count if object_count else math.nan
            rows.append(
                {
                    "feature_class": class_name,
                    "source_curvature_bin": str(row["curvature_bin"]),
                    "object_count": object_count,
                    "raw_volume_mm3": volume_mm3,
                    "available_path_length_mm": float(row["available_path_length_mm"]),
                    "path_length_share": float(row["path_length_share"]),
                    "raw_volume_share": float(row["raw_volume_share"]),
                    "raw_count_share": float(row["raw_count_share"]),
                    "volume_per_path_mm3_per_mm": float(row["volume_rate"]),
                    "count_per_path_per_mm": float(row["count_rate"]),
                    "q_V": float(row["volume_rate"] / volume_rate_sum),
                    "q_N": float(row["count_rate"] / count_rate_sum),
                    "E_V": float(row["volume_share_to_length_share"]),
                    "E_N": float(row["count_share_to_length_share"]),
                    "mean_object_volume_mm3": float(mean_volume),
                    "mean_object_volume_ratio_to_class": float(
                        mean_volume / class_mean_volume
                    ),
                    "mechanical_source": relative_metadata(str(OLD_NORMALIZATION_PATH)),
                }
            )
    return pd.DataFrame(rows)


def _build_radial_table(
    mappings: dict[str, pd.DataFrame],
    exposure: pd.DataFrame,
    resolved_schemes: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    segment_length = exposure["segment_length_mm"].to_numpy(dtype=float)
    curve_radius = exposure["r_mm"].to_numpy(dtype=float)
    rows: list[dict[str, float | int | str | bool]] = []
    for scheme_name, (lower_mm, upper_mm) in resolved_schemes.items():
        curve_inner = (curve_radius >= lower_mm) & (curve_radius < upper_mm)
        curve_masks = {
            "inner": curve_inner,
            "rest": ~curve_inner,
        }
        path_lengths = {
            region: float(segment_length[mask].sum())
            for region, mask in curve_masks.items()
        }
        total_path_length = float(sum(path_lengths.values()))
        if total_path_length <= 0.0:
            raise ValueError(f"No fitted path exposure for radial scheme {scheme_name}")

        for class_name in CLASS_ORDER:
            table = mappings[class_name]
            object_radius = table["r_mm"].to_numpy(dtype=float)
            object_inner = (object_radius >= lower_mm) & (object_radius < upper_mm)
            object_masks = {
                "inner": object_inner,
                "rest": ~object_inner,
            }
            class_count = int(len(table))
            class_volume_voxel = float(table["volume_voxel"].sum())
            class_volume_mm3 = float(table["volume_mm3"].sum())
            class_mean_volume_mm3 = class_volume_mm3 / class_count
            for region in ["inner", "rest"]:
                mask = object_masks[region]
                object_count = int(mask.sum())
                volume_voxel = float(table.loc[mask, "volume_voxel"].sum())
                volume_mm3 = float(table.loc[mask, "volume_mm3"].sum())
                path_length_mm = path_lengths[region]
                path_share = path_length_mm / total_path_length
                volume_share = volume_mm3 / class_volume_mm3
                count_share = object_count / class_count
                mean_volume = volume_mm3 / object_count if object_count else math.nan
                geometric_proxy = (
                    path_length_mm
                    * ELECTRODE_THICKNESS_MM
                    * AXIAL_SLICE_COUNT
                    * PIXEL_SIZE_MM
                )
                rows.append(
                    {
                        "scheme": scheme_name,
                        "region": region,
                        "inner_lower_mm": lower_mm,
                        "inner_upper_mm": upper_mm,
                        "feature_class": class_name,
                        "object_count": object_count,
                        "raw_volume_voxel": volume_voxel,
                        "raw_volume_mm3": volume_mm3,
                        "raw_volume_share": volume_share,
                        "raw_count_share": count_share,
                        "available_path_length_mm": path_length_mm,
                        "path_length_share": path_share,
                        "volume_per_path_mm3_per_mm": volume_mm3 / path_length_mm,
                        "count_per_path_per_mm": object_count / path_length_mm,
                        "E_V": volume_share / path_share,
                        "E_N": count_share / path_share,
                        "mean_object_volume_mm3": mean_volume,
                        "mean_object_volume_ratio_to_class": (
                            mean_volume / class_mean_volume_mm3
                        ),
                        "geometric_proxy_volume_mm3": geometric_proxy,
                        "defect_volume_fraction_geometric_proxy": (
                            volume_mm3 / geometric_proxy
                        ),
                        "geometric_proxy_is_not_measured_cathode_volume": True,
                        "axial_slice_count": AXIAL_SLICE_COUNT,
                        "axial_span_mm": AXIAL_SLICE_COUNT * PIXEL_SIZE_MM,
                    }
                )
    return pd.DataFrame(rows)


def _build_curvature_sensitivity(
    mappings: dict[str, pd.DataFrame],
    exposure: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    curve_curvature = exposure["curvature_mm"].to_numpy(dtype=float)
    curve_segment_length = exposure["segment_length_mm"].to_numpy(dtype=float)
    for low in THRESHOLD_LOWS:
        for high in THRESHOLD_HIGHS:
            curve_bins = np.select(
                [curve_curvature < low, curve_curvature < high],
                ["< low", "low - high"],
                default=">= high",
            )
            path_by_bin = {
                label: float(curve_segment_length[curve_bins == label].sum())
                for label in CURVATURE_BIN_ORDER
            }
            total_path = float(sum(path_by_bin.values()))
            for class_name in CLASS_ORDER:
                table = mappings[class_name]
                object_curvature = table["curvature_1_per_mm"].to_numpy(dtype=float)
                object_bins = np.select(
                    [object_curvature < low, object_curvature < high],
                    ["< low", "low - high"],
                    default=">= high",
                )
                total_count = int(len(table))
                total_volume = float(table["volume_mm3"].sum())
                for label in CURVATURE_BIN_ORDER:
                    mask = object_bins == label
                    object_count = int(mask.sum())
                    volume_mm3 = float(table.loc[mask, "volume_mm3"].sum())
                    path_length = path_by_bin[label]
                    volume_share = volume_mm3 / total_volume
                    count_share = object_count / total_count
                    path_share = path_length / total_path
                    rows.append(
                        {
                            "low_threshold_1_per_mm": low,
                            "high_threshold_1_per_mm": high,
                            "curvature_bin": label,
                            "feature_class": class_name,
                            "object_count": object_count,
                            "raw_volume_mm3": volume_mm3,
                            "raw_volume_share": volume_share,
                            "raw_count_share": count_share,
                            "available_path_length_mm": path_length,
                            "path_length_share": path_share,
                            "E_V": volume_share / path_share,
                            "E_N": count_share / path_share,
                            "volume_per_path_mm3_per_mm": (volume_mm3 / path_length),
                            "count_per_path_per_mm": object_count / path_length,
                            "boundary_scenario_only": True,
                            "no_confidence_interval_or_threshold_selection_claim": True,
                        }
                    )
    return pd.DataFrame(rows)


def _build_arc_angular_exposure(
    exposure: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    x_values = exposure["x"].to_numpy(dtype=float)
    y_values = exposure["y"].to_numpy(dtype=float)
    segment_length = exposure["segment_length_mm"].to_numpy(dtype=float)
    radius_floor = np.floor(
        np.hypot(
            x_values - ANGLE_CENTER_X_PIXEL,
            y_values - ANGLE_CENTER_Y_PIXEL,
        )
    ).astype(np.int32)
    angle_deg = np.mod(
        np.degrees(
            np.arctan2(
                ANGLE_CENTER_Y_PIXEL - y_values,
                x_values - ANGLE_CENTER_X_PIXEL,
            )
        ),
        360.0,
    )
    annulus = (radius_floor >= ANGLE_RADIAL_FLOOR_MIN) & (
        radius_floor < ANGLE_RADIAL_FLOOR_MAX_EXCLUSIVE
    )
    sector = (
        annulus
        & (angle_deg >= ANGLE_SECTOR_LOW_DEG)
        & (angle_deg < ANGLE_SECTOR_HIGH_DEG)
    )
    masks = {
        "sector": sector,
        "complement": annulus & ~sector,
    }
    result: dict[str, dict[str, float | int]] = {}
    total_path = float(segment_length[annulus].sum())
    for region, mask in masks.items():
        result[region] = {
            "arc_point_count": int(mask.sum()),
            "arc_path_length_mm": float(segment_length[mask].sum()),
            "arc_path_share": float(segment_length[mask].sum() / total_path),
        }
    result["total"] = {
        "arc_point_count": int(annulus.sum()),
        "arc_path_length_mm": total_path,
        "arc_path_share": 1.0,
    }
    return result


def _write_curve_parameter_table(
    curve_parameters: dict[str, float | str],
    output_path: Path,
) -> None:
    rows = []
    units = {
        "fit_a_mm": "mm",
        "fit_b_mm": "mm/rad",
        "r_min_mm": "mm",
        "r_last_mm": "mm",
        "theta_min_rad": "rad",
        "theta_max_rad": "rad",
        "pitch_2pi_b_mm": "mm",
        "first_observed_turn_upper_mm": "mm",
        "radius_at_kappa_0.2_mm": "mm",
        "radius_at_kappa_0.3_mm": "mm",
        "kappa_at_r_2_mm_inverse": "1/mm",
    }
    for key, value in curve_parameters.items():
        if isinstance(value, str):
            continue
        rows.append(
            {
                "parameter": key,
                "value": value,
                "units": units.get(key, ""),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)
