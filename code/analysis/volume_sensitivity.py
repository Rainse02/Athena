"""Object-volume cutoff sensitivity using the final fixed-center mapping tables."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

from package_paths import OUTPUT_ROOT as ANALYSIS_OUTPUT_ROOT, relative_metadata


INPUT_ROOT = ANALYSIS_OUTPUT_ROOT / "fixed_center" / "curvature"
OUTPUT_ROOT = ANALYSIS_OUTPUT_ROOT / "fixed_center" / "resolution"

PIXEL_SIZE_MM = 0.01684976915031592

VOXEL_VOLUME_MM3 = PIXEL_SIZE_MM**3

THRESHOLDS = (1, 8, 27, 64, 125, 216, 512, 1000)

CLASS_FILES = {
    "Detach": "Detach_object_mapping.csv",
    "Internal": "Internal_object_mapping.csv",
    "Surface": "Surface_object_mapping.csv",
}

EXPECTED_BASELINE_COUNTS = {
    "Detach": 5492,
    "Internal": 751,
    "Surface": 6377,
}

CURVATURE_BIN_ORDER = ["< 0.2", "0.2 - 0.3", ">= 0.3"]

CURVATURE_BIN_LABELS = {
    "< 0.2": "< 0.2 mm⁻¹",
    "0.2 - 0.3": "0.2–0.3 mm⁻¹",
    ">= 0.3": "≥ 0.3 mm⁻¹",
}

CURVATURE_BIN_COLOURS = {
    "< 0.2": "#0072B2",
    "0.2 - 0.3": "#D55E00",
    ">= 0.3": "#009E73",
}

CLASS_COLOURS = {
    "Detach": "#D55E00",
    "Internal": "#0072B2",
    "Surface": "#009E73",
}

MAPPING_REQUIRED_COLUMNS = {
    "feature_class",
    "object_index",
    "volume_voxel",
    "volume_mm3",
    "curvature_bin",
}

EXPOSURE_REQUIRED_COLUMNS = {"curvature_bin", "segment_length_mm"}

QUANTILE_LEVELS = {
    "min": None,
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
    "max": None,
}


def _require_columns(
    table: pd.DataFrame,
    required: set[str],
    path: Path,
) -> None:
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _coerce_finite_numeric(
    table: pd.DataFrame,
    columns: list[str],
    path: Path,
) -> None:
    for column in columns:
        try:
            table[column] = pd.to_numeric(table[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} has non-numeric values in {column}") from exc
        values = table[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path} has non-finite values in {column}")


def _read_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    mappings: dict[str, pd.DataFrame] = {}
    for class_name, filename in CLASS_FILES.items():
        path = INPUT_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pd.read_csv(path)
        _require_columns(table, MAPPING_REQUIRED_COLUMNS, path)
        _coerce_finite_numeric(table, ["volume_voxel", "volume_mm3"], path)

        observed_classes = set(table["feature_class"].astype(str).unique())
        if observed_classes != {class_name}:
            raise ValueError(
                f"{path} has feature_class values {sorted(observed_classes)}; "
                f"expected only {class_name}"
            )
        table["curvature_bin"] = table["curvature_bin"].astype(str)
        observed_bins = set(table["curvature_bin"].unique())
        if observed_bins != set(CURVATURE_BIN_ORDER):
            raise ValueError(
                f"{path} has curvature bins {sorted(observed_bins)}; "
                f"expected {CURVATURE_BIN_ORDER}"
            )
        if (table["volume_voxel"] <= 0).any():
            raise ValueError(f"{path} contains non-positive volume_voxel values")
        if (table["volume_mm3"] <= 0).any():
            raise ValueError(f"{path} contains non-positive volume_mm3 values")

        expected_volume_mm3 = table["volume_voxel"] * VOXEL_VOLUME_MM3
        if not np.allclose(
            table["volume_mm3"].to_numpy(dtype=float),
            expected_volume_mm3.to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-12,
        ):
            raise ValueError(
                f"{path} volume_mm3 is inconsistent with the configured isotropic "
                "voxel spacing"
            )
        mappings[class_name] = table

    exposure_path = INPUT_ROOT / "fitted_curve_exposure.csv"
    if not exposure_path.is_file():
        raise FileNotFoundError(exposure_path)
    exposure = pd.read_csv(exposure_path)
    _require_columns(exposure, EXPOSURE_REQUIRED_COLUMNS, exposure_path)
    _coerce_finite_numeric(exposure, ["segment_length_mm"], exposure_path)
    exposure["curvature_bin"] = exposure["curvature_bin"].astype(str)
    observed_bins = set(exposure["curvature_bin"].unique())
    if observed_bins != set(CURVATURE_BIN_ORDER):
        raise ValueError(
            f"{exposure_path} has curvature bins {sorted(observed_bins)}; "
            f"expected {CURVATURE_BIN_ORDER}"
        )
    if (exposure["segment_length_mm"] < 0).any():
        raise ValueError(f"{exposure_path} contains negative segment lengths")

    exposure_by_bin = (
        exposure.groupby("curvature_bin", sort=False)["segment_length_mm"]
        .sum()
        .reindex(CURVATURE_BIN_ORDER)
    )
    if exposure_by_bin.isna().any() or (exposure_by_bin <= 0).any():
        raise ValueError(
            f"{exposure_path} must provide positive exposure for all curvature bins"
        )
    total_path_length = float(exposure_by_bin.sum())
    exposure_table = exposure_by_bin.rename("full_path_length_mm").reset_index()
    exposure_table["full_path_length_share"] = (
        exposure_table["full_path_length_mm"] / total_path_length
    )
    return mappings, exposure_table


def _summary_statistics(values: pd.Series, prefix: str) -> dict[str, float]:
    statistics: dict[str, float] = {}
    for statistic, level in QUANTILE_LEVELS.items():
        if statistic == "min":
            value = values.min()
        elif statistic == "max":
            value = values.max()
        else:
            value = values.quantile(float(level))
        statistics[f"{prefix}_{statistic}"] = float(value)
    return statistics


def _equivalent_sphere_diameter_mm(volume_mm3: pd.Series | np.ndarray) -> np.ndarray:
    volume_array = np.asarray(volume_mm3, dtype=float)
    return np.cbrt(6.0 * volume_array / math.pi)


def _build_inventory(mappings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for class_name, table in mappings.items():
        sphere_diameter_mm = _equivalent_sphere_diameter_mm(table["volume_mm3"])
        row: dict[str, float | int | str] = {
            "feature_class": class_name,
            "object_count": int(len(table)),
        }
        row.update(_summary_statistics(table["volume_voxel"], "volume_voxel"))
        row.update(_summary_statistics(table["volume_mm3"], "volume_mm3"))
        row.update(
            _summary_statistics(
                pd.Series(sphere_diameter_mm),
                "equivalent_sphere_diameter_mm",
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_scales() -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for threshold in THRESHOLDS:
        cube_length_mm = PIXEL_SIZE_MM * threshold ** (1.0 / 3.0)
        cube_volume_mm3 = threshold * VOXEL_VOLUME_MM3
        sphere_diameter_mm = PIXEL_SIZE_MM * (6.0 * threshold / math.pi) ** (1.0 / 3.0)
        rows.append(
            {
                "threshold_native_voxels": threshold,
                "threshold_condition": "volume_voxel >= threshold_native_voxels",
                "threshold_cube_length_mm": cube_length_mm,
                "threshold_cube_volume_mm3": cube_volume_mm3,
                "threshold_equivalent_sphere_diameter_mm": sphere_diameter_mm,
                "equivalent_sphere_is_volume_derived_proxy": True,
                "threshold_is_validated_minimum_size": False,
            }
        )
    return pd.DataFrame(rows)


def _build_threshold_table(
    mappings: dict[str, pd.DataFrame],
    scales: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    scales_by_threshold = scales.set_index("threshold_native_voxels")
    for class_name, table in mappings.items():
        baseline_count = int(len(table))
        baseline_volume_voxel = float(table["volume_voxel"].sum())
        baseline_volume_mm3 = float(table["volume_mm3"].sum())
        for threshold in THRESHOLDS:
            retained_mask = table["volume_voxel"] >= threshold
            retained = table.loc[retained_mask]
            retained_count = int(len(retained))
            retained_volume_voxel = float(retained["volume_voxel"].sum())
            retained_volume_mm3 = float(retained["volume_mm3"].sum())
            scale = scales_by_threshold.loc[threshold]
            rows.append(
                {
                    "feature_class": class_name,
                    "threshold_native_voxels": threshold,
                    "threshold_condition": "volume_voxel >= threshold_native_voxels",
                    "threshold_cube_length_mm": float(
                        scale["threshold_cube_length_mm"]
                    ),
                    "threshold_equivalent_sphere_diameter_mm": float(
                        scale["threshold_equivalent_sphere_diameter_mm"]
                    ),
                    "baseline_object_count": baseline_count,
                    "retained_object_count": retained_count,
                    "retained_object_fraction": retained_count / baseline_count,
                    "baseline_volume_voxel": baseline_volume_voxel,
                    "retained_volume_voxel": retained_volume_voxel,
                    "retained_volume_voxel_fraction": (
                        retained_volume_voxel / baseline_volume_voxel
                    ),
                    "baseline_volume_mm3": baseline_volume_mm3,
                    "retained_volume_mm3": retained_volume_mm3,
                    "retained_volume_mm3_fraction": (
                        retained_volume_mm3 / baseline_volume_mm3
                    ),
                    "threshold_is_sensitivity_screen": True,
                }
            )
    return pd.DataFrame(rows)


def _build_curvature_table(
    mappings: dict[str, pd.DataFrame],
    exposure_table: pd.DataFrame,
) -> pd.DataFrame:
    exposure_by_bin = exposure_table.set_index("curvature_bin")
    rows: list[dict[str, float | int | str | bool]] = []
    for class_name, table in mappings.items():
        baseline_volume_mm3 = float(table["volume_mm3"].sum())
        baseline_by_bin = (
            table.groupby("curvature_bin", sort=False)["volume_mm3"]
            .sum()
            .reindex(CURVATURE_BIN_ORDER, fill_value=0.0)
        )
        baseline_count_by_bin = (
            table.groupby("curvature_bin", sort=False)
            .size()
            .reindex(CURVATURE_BIN_ORDER, fill_value=0)
        )
        for threshold in THRESHOLDS:
            retained = table.loc[table["volume_voxel"] >= threshold]
            retained_volume_by_bin = (
                retained.groupby("curvature_bin", sort=False)["volume_mm3"]
                .sum()
                .reindex(CURVATURE_BIN_ORDER, fill_value=0.0)
            )
            retained_count_by_bin = (
                retained.groupby("curvature_bin", sort=False)
                .size()
                .reindex(CURVATURE_BIN_ORDER, fill_value=0)
            )
            retained_volume_mm3 = float(retained_volume_by_bin.sum())
            for curvature_bin in CURVATURE_BIN_ORDER:
                full_path_length_mm = float(
                    exposure_by_bin.loc[curvature_bin, "full_path_length_mm"]
                )
                full_path_length_share = float(
                    exposure_by_bin.loc[curvature_bin, "full_path_length_share"]
                )
                retained_volume_bin_mm3 = float(
                    retained_volume_by_bin.loc[curvature_bin]
                )
                retained_volume_share = (
                    retained_volume_bin_mm3 / retained_volume_mm3
                    if retained_volume_mm3 > 0
                    else float("nan")
                )
                rows.append(
                    {
                        "feature_class": class_name,
                        "threshold_native_voxels": threshold,
                        "threshold_condition": "volume_voxel >= threshold_native_voxels",
                        "curvature_bin": curvature_bin,
                        "full_path_length_mm": full_path_length_mm,
                        "full_path_length_share": full_path_length_share,
                        "baseline_bin_object_count": int(
                            baseline_count_by_bin.loc[curvature_bin]
                        ),
                        "baseline_bin_volume_mm3": float(
                            baseline_by_bin.loc[curvature_bin]
                        ),
                        "baseline_bin_volume_share": float(
                            baseline_by_bin.loc[curvature_bin] / baseline_volume_mm3
                        ),
                        "retained_bin_object_count": int(
                            retained_count_by_bin.loc[curvature_bin]
                        ),
                        "retained_bin_volume_mm3": retained_volume_bin_mm3,
                        "retained_bin_volume_share_within_threshold": retained_volume_share,
                        "retained_bin_volume_share_of_class_baseline": (
                            retained_volume_bin_mm3 / baseline_volume_mm3
                        ),
                        "retained_bin_volume_fraction_of_class_baseline": (
                            retained_volume_bin_mm3 / baseline_volume_mm3
                        ),
                        "volume_enrichment_vs_full_path": (
                            retained_volume_share / full_path_length_share
                        ),
                        "path_denominator_uses_all_fitted_exposure_segments": True,
                        "threshold_is_sensitivity_screen": True,
                    }
                )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    png_path = OUTPUT_ROOT / f"{stem}.png"
    svg_path = OUTPUT_ROOT / f"{stem}.svg"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    paths["png"] = str(png_path)
    paths["svg"] = str(svg_path)
    return paths


def _configure_threshold_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(THRESHOLDS))
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.tick_params(axis="x", labelrotation=35)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)


def _make_retention_plot(threshold_table: pd.DataFrame) -> dict[str, str]:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.8),
        sharex=True,
        constrained_layout=True,
    )
    for class_name in CLASS_FILES:
        subset = threshold_table[
            threshold_table["feature_class"] == class_name
        ].sort_values("threshold_native_voxels")
        colour = CLASS_COLOURS[class_name]
        axes[0].plot(
            subset["threshold_native_voxels"],
            subset["retained_object_fraction"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=colour,
            label=class_name,
        )
        axes[1].plot(
            subset["threshold_native_voxels"],
            subset["retained_volume_mm3_fraction"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=colour,
            label=class_name,
        )

    axes[0].set_title("Object-count retention")
    axes[1].set_title("Physical-volume retention")
    axes[0].set_ylabel("Retained fraction of class baseline")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_ylim(0.0, 1.05)
    for axis in axes:
        _configure_threshold_axis(axis)
        axis.set_xlabel("Minimum object volume (native voxels; ≥ threshold)")
    axes[1].legend(frameon=False, loc="lower left")
    figure.suptitle("R3C4 native voxel-volume sensitivity screens", fontsize=13)
    figure.text(
        0.5,
        -0.03,
        "Thresholds are descriptive sensitivity screens, not validated minimum sizes.",
        ha="center",
        fontsize=9,
    )
    paths = _save_figure(figure, "volume_threshold_retention")
    plt.close(figure)
    return paths


def _make_enrichment_plot(curvature_table: pd.DataFrame) -> dict[str, str]:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    y_max = float(curvature_table["volume_enrichment_vs_full_path"].max())
    for axis, class_name in zip(axes, CLASS_FILES):
        subset = curvature_table[curvature_table["feature_class"] == class_name]
        for curvature_bin in CURVATURE_BIN_ORDER:
            bin_subset = subset[subset["curvature_bin"] == curvature_bin].sort_values(
                "threshold_native_voxels"
            )
            axis.plot(
                bin_subset["threshold_native_voxels"],
                bin_subset["volume_enrichment_vs_full_path"],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=CURVATURE_BIN_COLOURS[curvature_bin],
                label=CURVATURE_BIN_LABELS[curvature_bin],
            )
        axis.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--")
        axis.set_title(class_name)
        _configure_threshold_axis(axis)
        axis.set_xlabel("Minimum object volume (native voxels; ≥ threshold)")
        axis.set_ylim(0.0, y_max * 1.08)
    axes[0].set_ylabel("Retained defect-volume share / full path-length share")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    figure.suptitle(
        "Curvature-bin volume enrichment under native volume screens",
        fontsize=13,
    )
    figure.text(
        0.5,
        -0.03,
        "The denominator is the same full fitted-curve exposure at every threshold.",
        ha="center",
        fontsize=9,
    )
    paths = _save_figure(figure, "curvature_volume_enrichment")
    plt.close(figure)
    return paths


def _run_checks(
    mappings: dict[str, pd.DataFrame],
    exposure_table: pd.DataFrame,
    threshold_table: pd.DataFrame,
    curvature_table: pd.DataFrame,
) -> dict[str, bool | float]:
    checks: dict[str, bool | float] = {}
    for class_name, table in mappings.items():
        observed_count = int(len(table))
        expected_count = EXPECTED_BASELINE_COUNTS[class_name]
        checks[f"{class_name}_baseline_count_matches_expected"] = (
            observed_count == expected_count
        )

        class_thresholds = threshold_table[
            threshold_table["feature_class"] == class_name
        ].sort_values("threshold_native_voxels")
        count_fraction = class_thresholds["retained_object_fraction"].to_numpy(
            dtype=float
        )
        volume_fraction = class_thresholds["retained_volume_mm3_fraction"].to_numpy(
            dtype=float
        )
        checks[f"{class_name}_count_fraction_monotonic_nonincreasing"] = bool(
            np.all(np.diff(count_fraction) <= 1e-12)
        )
        checks[f"{class_name}_volume_fraction_monotonic_nonincreasing"] = bool(
            np.all(np.diff(volume_fraction) <= 1e-12)
        )

        threshold_one = class_thresholds.iloc[0]
        checks[f"{class_name}_threshold_1_count_is_baseline"] = bool(
            threshold_one["retained_object_count"] == observed_count
        )
        checks[f"{class_name}_threshold_1_volume_is_baseline"] = bool(
            np.isclose(
                threshold_one["retained_volume_mm3"],
                table["volume_mm3"].sum(),
                rtol=1e-12,
                atol=1e-15,
            )
        )

        for threshold in THRESHOLDS:
            threshold_curvature = curvature_table[
                (curvature_table["feature_class"] == class_name)
                & (curvature_table["threshold_native_voxels"] == threshold)
            ]
            retained_volume = float(
                class_thresholds.loc[
                    class_thresholds["threshold_native_voxels"] == threshold,
                    "retained_volume_mm3",
                ].iloc[0]
            )
            grouped_volume = float(threshold_curvature["retained_bin_volume_mm3"].sum())
            checks[
                f"{class_name}_threshold_{threshold}_grouped_volume_matches_total"
            ] = bool(
                np.isclose(
                    grouped_volume,
                    retained_volume,
                    rtol=1e-12,
                    atol=1e-15,
                )
            )

    path_share_sum = float(exposure_table["full_path_length_share"].sum())
    checks["full_path_exposure_share_sum_is_one"] = bool(
        np.isclose(path_share_sum, 1.0, rtol=1e-12, atol=1e-15)
    )
    checks["all_acceptance_checks_pass"] = all(bool(value) for value in checks.values())
    if not checks["all_acceptance_checks_pass"]:
        failed = [key for key, value in checks.items() if not bool(value)]
        raise AssertionError(f"Resolution-volume acceptance checks failed: {failed}")
    return checks


def _format_number(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def _write_notes(
    inventory: pd.DataFrame,
    threshold_table: pd.DataFrame,
    curvature_table: pd.DataFrame,
    exposure_table: pd.DataFrame,
    checks: dict[str, bool | float],
) -> Path:
    lines = [
        "# R3C4 defect-object volume sensitivity analysis",
        "",
        "This analysis is a descriptive native-voxel-volume sensitivity screen "
        "for the cached R3C4 curvature mapping tables. The strict retention "
        "condition is `volume_voxel >= threshold_native_voxels`.",
        "",
        "## Scope and definitions",
        "",
        f"- Isotropic sampling used for physical conversion: "
        f"{PIXEL_SIZE_MM:.16g} mm; one voxel is "
        f"{VOXEL_VOLUME_MM3:.16g} mm³.",
        "- Thresholds are native voxel counts: "
        + ", ".join(str(value) for value in THRESHOLDS)
        + ". They are sensitivity screens, not validated minimum object sizes.",
        "- Cube-length and equivalent-sphere diameter columns are geometric "
        "scales associated with each threshold.",
        "- Equivalent-sphere diameter is derived from object volume with "
        "`d = (6V/π)^(1/3)`; it is not an aperture, detection threshold, or "
        "independent size measurement.",
        "- Curvature enrichment uses each retained class's volume share divided "
        "by the corresponding share of the same full fitted-curve exposure "
        "table. The exposure denominator is not filtered by defects.",
        "- No source CT was reopened, no label rescan or resampling was "
        "performed, and no detection calibration was attempted.",
        "- The analysis does not infer true missed-defect counts or actual "
        "imaging resolution.",
        "",
        "## Baseline inventory",
        "",
        "| Class | Objects | Voxel p50 | Voxel p95 | Physical p50 (mm³) | "
        "Physical p95 (mm³) | Sphere p50 (mm) | Sphere p95 (mm) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for class_name in CLASS_FILES:
        row = inventory[inventory["feature_class"] == class_name].iloc[0]
        lines.append(
            f"| {class_name} | {int(row['object_count'])} | "
            f"{_format_number(float(row['volume_voxel_p50']))} | "
            f"{_format_number(float(row['volume_voxel_p95']))} | "
            f"{_format_number(float(row['volume_mm3_p50']), 5)} | "
            f"{_format_number(float(row['volume_mm3_p95']), 5)} | "
            f"{_format_number(float(row['equivalent_sphere_diameter_mm_p50']), 5)} | "
            f"{_format_number(float(row['equivalent_sphere_diameter_mm_p95']), 5)} |"
        )

    lines.extend(
        [
            "",
            "## Selected retention results",
            "",
            "Fractions below are relative to the unfiltered class baseline. "
            "The complete threshold-by-class table is in "
            "`threshold_sensitivity.csv`.",
            "",
            "| Threshold (voxels) | Cube scale (mm) | Sphere scale (mm) | "
            "Class | Count fraction | Physical-volume fraction |",
            "| ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    selected_thresholds = (27, 1000)
    for threshold in selected_thresholds:
        scale = _threshold_scales()
        scale_row = scale[scale["threshold_native_voxels"] == threshold].iloc[0]
        for class_name in CLASS_FILES:
            row = threshold_table[
                (threshold_table["feature_class"] == class_name)
                & (threshold_table["threshold_native_voxels"] == threshold)
            ].iloc[0]
            lines.append(
                f"| {threshold} | "
                f"{_format_number(float(scale_row['threshold_cube_length_mm']), 5)} | "
                f"{_format_number(float(scale_row['threshold_equivalent_sphere_diameter_mm']), 5)} | "
                f"{class_name} | "
                f"{_format_number(float(row['retained_object_fraction']), 4)} | "
                f"{_format_number(float(row['retained_volume_mm3_fraction']), 4)} |"
            )

    lines.extend(
        [
            "",
            "## Full-path curvature exposure",
            "",
            "| Curvature bin | Full fitted-path length (mm) | Path share |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in exposure_table.iterrows():
        lines.append(
            f"| {row['curvature_bin']} | "
            f"{_format_number(float(row['full_path_length_mm']), 6)} | "
            f"{_format_number(float(row['full_path_length_share']), 5)} |"
        )

    lines.extend(
        [
            "",
            "At the 1-voxel baseline, the curvature-bin volume enrichments "
            "relative to these full-path shares are:",
            "",
            "| Class | < 0.2 | 0.2 - 0.3 | >= 0.3 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for class_name in CLASS_FILES:
        rows = curvature_table[
            (curvature_table["feature_class"] == class_name)
            & (curvature_table["threshold_native_voxels"] == 1)
        ].set_index("curvature_bin")
        lines.append(
            f"| {class_name} | "
            f"{_format_number(float(rows.loc['< 0.2', 'volume_enrichment_vs_full_path']), 4)} | "
            f"{_format_number(float(rows.loc['0.2 - 0.3', 'volume_enrichment_vs_full_path']), 4)} | "
            f"{_format_number(float(rows.loc['>= 0.3', 'volume_enrichment_vs_full_path']), 4)} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance checks",
            "",
            "The expected baseline counts, non-increasing retained fractions, "
            "threshold-1 baseline equivalence, grouped-volume sums, and full "
            "path-share sum all passed.",
            "",
            f"`all_acceptance_checks_pass`: `{bool(checks['all_acceptance_checks_pass'])}`",
            "",
            "## Output files",
            "",
            "- `object_volume_inventory.csv`: per-class min, p10, p25, p50, "
            "p75, p90, p95, p99, and max for native voxel volume, physical "
            "volume, and volume-derived sphere diameter.",
            "- `threshold_scales.csv`: threshold cube and equivalent-sphere " "scales.",
            "- `threshold_sensitivity.csv`: class-level retained count and "
            "volume totals/fractions for every threshold.",
            "- `curvature_volume_enrichment.csv`: retained curvature-bin volume "
            "shares and full-path enrichment for every class and threshold.",
            "- `volume_threshold_retention.png` and `.svg`: count and physical "
            "volume retention plots.",
            "- `curvature_volume_enrichment.png` and `.svg`: curvature-bin "
            "volume enrichment plots.",
            "- `metadata.json`: inputs, formulas, operation limits, environment, "
            "and acceptance-check results.",
        ]
    )
    notes_path = OUTPUT_ROOT / "resolution_volume_analysis_notes.md"
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return notes_path


def _write_metadata(
    inventory: pd.DataFrame,
    threshold_table: pd.DataFrame,
    curvature_table: pd.DataFrame,
    exposure_table: pd.DataFrame,
    scales: pd.DataFrame,
    checks: dict[str, bool | float],
    plot_paths: dict[str, dict[str, str]],
    notes_path: Path,
) -> Path:
    totals: dict[str, dict[str, float | int]] = {}
    for class_name, table in (
        (class_name, pd.read_csv(INPUT_ROOT / filename))
        for class_name, filename in CLASS_FILES.items()
    ):
        totals[class_name] = {
            "object_count": int(len(table)),
            "volume_voxel": float(table["volume_voxel"].sum()),
            "volume_mm3": float(table["volume_mm3"].sum()),
        }

    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "analysis": {
            "name": "object_volume_cutoff_sensitivity",
            "script": str(Path(__file__).resolve()),
            "generated_at_utc": generated_at,
        },
        "inputs": {
            "input_root": str(INPUT_ROOT),
            "mapping_files": {
                class_name: str(INPUT_ROOT / filename)
                for class_name, filename in CLASS_FILES.items()
            },
            "fitted_curve_exposure": str(INPUT_ROOT / "fitted_curve_exposure.csv"),
            "required_mapping_columns": sorted(MAPPING_REQUIRED_COLUMNS),
            "required_exposure_columns": sorted(EXPOSURE_REQUIRED_COLUMNS),
        },
        "sampling": {
            "isotropic_spacing_mm": PIXEL_SIZE_MM,
            "voxel_volume_mm3": VOXEL_VOLUME_MM3,
            "physical_volume_source": "volume_mm3 from cached mapping tables",
            "conversion_formula": "volume_voxel * isotropic_spacing_mm^3",
        },
        "thresholds": {
            "native_voxels": list(THRESHOLDS),
            "condition": "volume_voxel >= threshold_native_voxels",
            "strict_semantics": True,
            "interpretation": "descriptive sensitivity screens, not validated minimum sizes",
            "scales_csv": str(OUTPUT_ROOT / "threshold_scales.csv"),
        },
        "equivalent_sphere_proxy": {
            "formula": "d_mm = (6 * volume_mm3 / pi)^(1/3)",
            "meaning": "volume-derived sphere diameter proxy",
            "not_aperture_or_detection_threshold": True,
            "not_independent_shape_measurement": True,
        },
        "curvature_normalization": {
            "bin_order": CURVATURE_BIN_ORDER,
            "path_length_source": "sum of segment_length_mm by curvature_bin in fitted_curve_exposure.csv",
            "path_denominator_is_full_fitted_exposure": True,
            "path_denominator_is_not_filtered_by_defects": True,
            "enrichment_formula": "retained class volume share within bin / full fitted-path length share",
        },
        "operations": {
            "source_ct_reopened": False,
            "segmentation_rescanned": False,
            "resampling_performed": False,
            "detection_calibration_performed": False,
            "true_missed_defect_counts_inferred": False,
            "actual_imaging_resolution_inferred": False,
        },
        "environment": {
            "python_version": sys.version,
        },
        "baseline_totals": totals,
        "acceptance_checks": checks,
        "outputs": {
            "inventory_csv": str(OUTPUT_ROOT / "object_volume_inventory.csv"),
            "threshold_scales_csv": str(OUTPUT_ROOT / "threshold_scales.csv"),
            "threshold_sensitivity_csv": str(OUTPUT_ROOT / "threshold_sensitivity.csv"),
            "curvature_volume_enrichment_csv": str(
                OUTPUT_ROOT / "curvature_volume_enrichment.csv"
            ),
            "notes_md": str(notes_path),
            "plots": plot_paths,
        },
        "row_counts": {
            "inventory_rows": int(len(inventory)),
            "threshold_rows": int(len(threshold_table)),
            "curvature_rows": int(len(curvature_table)),
            "exposure_bins": int(len(exposure_table)),
            "threshold_scale_rows": int(len(scales)),
        },
    }
    metadata_path = OUTPUT_ROOT / "metadata.json"
    metadata_path.write_text(
        json.dumps(relative_metadata(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def main() -> int:
    mappings, exposure_table = _read_inputs()
    inventory = _build_inventory(mappings)
    scales = _threshold_scales()
    threshold_table = _build_threshold_table(mappings, scales)
    curvature_table = _build_curvature_table(mappings, exposure_table)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    checks = _run_checks(
        mappings,
        exposure_table,
        threshold_table,
        curvature_table,
    )

    inventory_path = OUTPUT_ROOT / "object_volume_inventory.csv"
    scales_path = OUTPUT_ROOT / "threshold_scales.csv"
    threshold_path = OUTPUT_ROOT / "threshold_sensitivity.csv"
    curvature_path = OUTPUT_ROOT / "curvature_volume_enrichment.csv"
    float_format = "%.15g"
    inventory.to_csv(inventory_path, index=False, float_format=float_format)
    scales.to_csv(scales_path, index=False, float_format=float_format)
    threshold_table.to_csv(threshold_path, index=False, float_format=float_format)
    curvature_table.to_csv(curvature_path, index=False, float_format=float_format)

    plot_paths = {
        "retention": _make_retention_plot(threshold_table),
        "curvature_enrichment": _make_enrichment_plot(curvature_table),
    }
    notes_path = _write_notes(
        inventory,
        threshold_table,
        curvature_table,
        exposure_table,
        checks,
    )
    metadata_path = _write_metadata(
        inventory,
        threshold_table,
        curvature_table,
        exposure_table,
        scales,
        checks,
        plot_paths,
        notes_path,
    )

    print(f"inventory: {inventory_path}")
    print(f"threshold scales: {scales_path}")
    print(f"threshold sensitivity: {threshold_path}")
    print(f"curvature enrichment: {curvature_path}")
    print(f"notes: {notes_path}")
    print(f"metadata: {metadata_path}")
    print(f"checks: {checks['all_acceptance_checks_pass']}")
    return 0
