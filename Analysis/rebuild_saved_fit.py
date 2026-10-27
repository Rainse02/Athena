"""Fit an Archimedean spiral to ordered contour points at a fixed center."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from settings import DATA_ROOT, OUTPUT_ROOT


CENTER = np.asarray([613.0, 603.5], dtype=float)


def spiral_func(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    """Return the Archimedean radius a + b*theta."""
    return a + b * np.asarray(theta, dtype=float)


def run() -> dict:
    """Convert contour points to polar coordinates and fit a and b."""
    table = pd.read_csv(DATA_ROOT / "curvature" / "contour_fit_points.csv")
    table = table.sort_values("point_index_contour_order")
    points = table[["x", "y"]].to_numpy(dtype=float)
    dx = points[:, 0] - CENTER[0]
    dy = -points[:, 1] + CENTER[1]
    radius = np.sqrt(dx**2 + dy**2)
    theta = np.unwrap(np.arctan2(dy, dx))
    parameters, covariance = curve_fit(spiral_func, theta, radius)
    source = json.loads(
        (DATA_ROOT / "curvature" / "fit_source_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    recorded = np.asarray([source["a_pixels"], source["b_pixels"]], dtype=float)
    summary = {
        "source": str(DATA_ROOT / "curvature" / "contour_fit_points.csv"),
        "input_level": "Saved extracted contour, preserving its point order",
        "center_pixels": CENTER.tolist(),
        "a_pixels": float(parameters[0]),
        "b_pixels": float(parameters[1]),
        "covariance_pixels": covariance.tolist(),
        "difference_from_recorded_parameters_pixels": (parameters - recorded).tolist(),
        "matches_recorded_fit": bool(
            np.allclose(parameters, recorded, rtol=1e-12, atol=1e-12)
        ),
        "point_count": len(table),
        "method": "Ordinary scipy.optimize.curve_fit with default settings; center fixed",
    }
    destination = OUTPUT_ROOT / "fixed_center" / "saved_fit_reconstruction.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["matches_recorded_fit"]:
        raise ValueError("The recorded contour did not reproduce the saved parameters.")
    print("Saved-contour fit:", destination)
    return summary


if __name__ == "__main__":
    run()
