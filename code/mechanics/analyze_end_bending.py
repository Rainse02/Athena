"""Reproduce final constrained/free mismatch and stored-base numerical statistics."""

from __future__ import annotations

import json
import numpy as np
from matplotlib.path import Path as PolygonPath

from metrics import (
    finite_percentile_record,
    interface_metrics,
    interval_mask,
    distance_to_interface,
)
from package_paths import DATA_ROOT, OUTPUT_ROOT, run_cli


DATA_DIR = DATA_ROOT
INPUT_DIR = DATA_DIR / "end_bending"
OUTPUT_DIR = OUTPUT_ROOT
SCALE = 1.65e6
CASES = (
    "F_end_bending_base",
    "F_zero_increment_check",
    "Figure6F_end_bending_mismatch",
    "F_end_bending_normal_free",
)


def surface_metrics(data: np.ndarray) -> dict:
    result = {}
    for label, side in (("inner", data[:, 1] < 0.2e-3), ("outer", data[:, 1] > 0.2e-3)):
        points = data[side]
        tangent = np.column_stack(
            (
                np.gradient(points[:, 0] + points[:, 2]),
                np.gradient(points[:, 1] + points[:, 3]),
            )
        )
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        tx, ty = tangent[:, 0], tangent[:, 1]
        stress = (
            points[:, 4] * tx**2 + 2 * points[:, 6] * tx * ty + points[:, 5] * ty**2
        ) / SCALE
        central = np.abs(points[:, 0]) <= 0.9e-3
        fit = np.polyfit(
            (points[central, 0] + points[central, 2]) * 1e3,
            (points[central, 1] + points[central, 3]) * 1e3,
            2,
        )
        result[label] = {
            "path_stress_signed_P05_P50_P95": [
                float(np.percentile(stress[central], p)) for p in (5, 50, 95)
            ],
            "path_tension_fraction": float(np.mean(stress[central] > 0)),
            "path_compression_fraction": float(np.mean(stress[central] < 0)),
            "reference_surface_normal_traction_abs_P95": float(
                np.percentile(np.abs(points[central, 5]), 95) / SCALE
            ),
            "reference_surface_tangent_traction_abs_P95": float(
                np.percentile(np.abs(points[central, 6]), 95) / SCALE
            ),
            "parabolic_fit_curvature_at_center_per_mm": float(
                -2 * fit[0] / (1 + fit[1] ** 2) ** 1.5
            ),
        }
    return result


def incremental_metrics(data: np.ndarray, factor: float) -> dict:
    upper = data[:, 1] > 0.2e-3
    core = np.abs(data[:, 0]) <= 1.2e-3
    delta = 0.005 * 0.488643305e-3
    q = np.clip((np.abs(data[:, 0]) - 1.2e-3) / 0.8e-3, 0, 1)
    window = 1 - 10 * q**3 + 15 * q**4 - 6 * q**5
    expected = factor * delta * window * upper
    dx = data[:, 2] - data[:, 4]
    dy = data[:, 3] - data[:, 5]
    ends = np.abs(np.abs(data[:, 0]) - 2e-3) < 1e-12
    return {
        "x_increment_prescription_max_abs_error_m": float(
            np.max(np.abs(dx - expected))
        ),
        "y_increment_max_abs_m": float(np.max(np.abs(dy))),
        "upper_core_x_increment_median_um": float(np.median(dx[upper & core]) * 1e6),
        "lower_core_x_increment_median_um": float(np.median(dx[~upper & core]) * 1e6),
        "end_corner_x_increment_max_abs_m": float(np.max(np.abs(dx[ends]))),
    }


def main() -> None:
    config = json.loads((DATA_DIR / "geometry_config.json").read_text(encoding="utf-8"))
    base = np.loadtxt(INPUT_DIR / "F_end_bending_base_matrix.txt", comments="%")
    coordinates_mm = base[:, :2] * 1e3
    is_metal = PolygonPath(config["vertices_mm"]).contains_points(coordinates_mm)
    distance = distance_to_interface(
        coordinates_mm[:, 0], coordinates_mm[:, 1], config["vertices_mm"]
    )
    near = ~is_metal & (distance <= config["matrix_neighborhood_distance_mm"])
    fields = {}
    summary = {
        "normalization_Pa": SCALE,
        "curvature_parameter_per_mm": 0.01,
        "full_core_path_increment_um": 2.443216525,
        "cases": {},
    }
    for name in CASES:
        data = np.loadtxt(INPUT_DIR / f"{name}_matrix.txt", comments="%")
        if not np.array_equal(data[:, :2], base[:, :2]):
            raise ValueError("Matrix sample coordinates differ")
        fields[name] = data
        valid = ~is_metal & np.all(np.isfinite(data), axis=1)
        record = {"matrix": {}}
        for region, key in (
            ("node", "node_x_intervals_mm"),
            ("ordinary", "ordinary_x_intervals_mm"),
        ):
            mask = valid & near & interval_mask(coordinates_mm[:, 0], config[key])
            record["matrix"][region] = finite_percentile_record(data[:, 5], None, mask)
        record["matrix"]["node_to_ordinary_P95"] = (
            record["matrix"]["node"]["p95"] / record["matrix"]["ordinary"]["p95"]
        )
        record["matrix"]["display_quantiles"] = {
            str(p): float(np.percentile(data[valid, 5], p))
            for p in (0, 5, 50, 95, 99, 99.9, 100)
        }
        record["matrix"]["fraction_above_2p5"] = float(np.mean(data[valid, 5] > 2.5))
        interface = np.loadtxt(INPUT_DIR / f"{name}_interface.txt", comments="%")
        record["interface"] = interface_metrics(interface, config)
        surfaces = np.loadtxt(INPUT_DIR / f"{name}_surfaces.txt", comments="%")
        record["surfaces"] = surface_metrics(surfaces)
        increments = np.loadtxt(INPUT_DIR / f"{name}_increments.txt", comments="%")
        factor = 0 if name in CASES[:2] else 1
        record["increment_check"] = incremental_metrics(increments, factor)
        summary["cases"][name] = record

    replay = fields["F_zero_increment_check"]
    valid = (
        ~is_metal
        & np.all(np.isfinite(base), axis=1)
        & np.all(np.isfinite(replay), axis=1)
    )
    stress_difference = (replay[valid, 2:5] - base[valid, 2:5]) / SCALE
    summary["zero_increment_replay"] = {
        "stress_component_RMS_difference_normalized": float(
            np.sqrt(np.mean(stress_difference**2))
        ),
        "stress_component_max_abs_difference_normalized": float(
            np.max(np.abs(stress_difference))
        ),
        "displacement_max_abs_difference_m": float(
            np.max(np.abs(replay[valid, 6:8] - base[valid, 6:8]))
        ),
    }
    for name in CASES[2:]:
        current = fields[name]
        valid = (
            ~is_metal
            & np.all(np.isfinite(base), axis=1)
            & np.all(np.isfinite(current), axis=1)
        )
        perturbation = (current[valid, 2:5] - base[valid, 2:5]) / SCALE
        summary["cases"][name]["stress_increment_RMS_from_base"] = float(
            np.sqrt(np.mean(perturbation**2))
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_case = fields["Figure6F_end_bending_mismatch"]
    valid = ~is_metal & np.all(np.isfinite(main_case), axis=1)
    axial = {}
    for region, key in (
        ("node", "node_x_intervals_mm"),
        ("ordinary", "ordinary_x_intervals_mm"),
    ):
        mask = valid & near & interval_mask(coordinates_mm[:, 0], config[key])
        axial[region] = {
            "axial_positive_P95": float(
                np.percentile(np.maximum(main_case[mask, 2] / SCALE, 0), 95)
            ),
            "axial_negative_magnitude_P95": float(
                np.percentile(np.maximum(-main_case[mask, 2] / SCALE, 0), 95)
            ),
            "principal_P95": float(np.percentile(main_case[mask, 5], 95)),
        }
    (OUTPUT_DIR / "current_axial_statistics.json").write_text(
        json.dumps(axial, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "end_bending_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    compact = {"zero_increment_replay": summary["zero_increment_replay"]}
    for name, result in summary["cases"].items():
        compact[name] = {
            "matrix": result["matrix"],
            "node_interface": result["interface"]["node"],
            "lower_neck_interface": result["interface"]["node_lower_neck"],
            "surfaces": result["surfaces"],
            "increment_check": result["increment_check"],
        }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    run_cli(main, __doc__)
