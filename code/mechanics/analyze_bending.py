"""Reproduce final bending surface P95 stresses and central circular fits."""

from __future__ import annotations

import json
import numpy as np

from package_paths import DATA_ROOT, OUTPUT_ROOT, run_cli


INPUT_DIR = DATA_ROOT / "bending"
OUTPUT_DIR = OUTPUT_ROOT
H = 0.488643305e-3


def summarize(name: str) -> dict:
    data = np.loadtxt(INPUT_DIR / f"{name}_surfaces.txt", comments="%")
    core = np.abs(data[:, 0]) <= 0.9e-3
    record = {}
    for label, y, sign in (("outer_tension", H, 1), ("inner_compression", 0, -1)):
        mask = core & np.isclose(data[:, 1], y, rtol=0, atol=1e-10)
        stress = np.sort(np.maximum(sign * data[mask, 4] / data[mask, 7], 0))
        p95 = float(stress[int(np.ceil(0.95 * stress.size)) - 1])
        x_deformed = (data[mask, 0] + data[mask, 2]) * 1e3
        y_deformed = (data[mask, 1] + data[mask, 3]) * 1e3
        # A single circle is a descriptive fit of the central surface, not an
        # assumption that the heterogeneous specimen has constant local curvature.
        matrix = np.column_stack(
            (2 * x_deformed, 2 * y_deformed, np.ones_like(x_deformed))
        )
        target = x_deformed**2 + y_deformed**2
        cx, cy, offset = np.linalg.lstsq(matrix, target, rcond=None)[0]
        radius = float(np.sqrt(offset + cx**2 + cy**2))
        distance = np.hypot(x_deformed - cx, y_deformed - cy)
        record[label] = {
            "normalized_P95": p95,
            "P95_MPa": float(p95 * data[0, 7] / 1e6),
            "surface_circle_fit_curvature_per_mm": 1 / radius,
            "circle_fit_radial_RMS_um": float(
                np.sqrt(np.mean((distance - radius) ** 2)) * 1e3
            ),
        }
    balance = np.loadtxt(INPUT_DIR / f"{name}_balance.txt", comments="%")
    record["final_right_reaction_N"] = float(balance[-1, 2])
    return record


def main() -> None:
    names = ("D_offset_buffer_k450", "D_reference_buffer_k450")
    result = {name: summarize(name) for name in names}
    result["offset_to_reference_outer_P95_ratio"] = (
        result[names[0]]["outer_tension"]["normalized_P95"]
        / result[names[1]]["outer_tension"]["normalized_P95"]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "bending_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_cli(main, __doc__)
