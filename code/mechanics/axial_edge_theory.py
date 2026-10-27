"""Evaluate the bonded-layer axial-edge solution in Supplementary Note S6.

This evaluator was added during archive preparation from the published-form
equations in the revised Supplementary Information. It is not a fit to the
measured Detachment profile. All three physical inputs must be supplied by the
caller. In particular, the axial transfer length is not inferred from the
separate winding-path model.

Equation: delta'' = delta / ell**2.
Boundary conditions: delta'(0) = delta'(H) = strain_mismatch.
Solution: delta = strain_mismatch * ell * sinh((y-H/2)/ell) / cosh(H/(2*ell)).
Coordinates, height, transfer length, and displacement are in millimetres.
The displacement gradient and strain mismatch are dimensionless.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from package_paths import OUTPUT_ROOT


def axial_edge_solution(
    y_mm: np.ndarray,
    height_mm: float,
    transfer_length_mm: float,
    strain_mismatch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relative displacement and its first axial derivative.

    The scaled exponential expressions below are algebraically equivalent to
    the hyperbolic functions and avoid overflow for a long layer.
    """
    parameters = np.asarray([height_mm, transfer_length_mm, strain_mismatch])
    if not np.all(np.isfinite(parameters)):
        raise ValueError("Physical parameters must be finite.")
    if height_mm <= 0 or transfer_length_mm <= 0:
        raise ValueError("Height and transfer length must be positive.")
    y_mm = np.asarray(y_mm, dtype=float)
    if not np.all(np.isfinite(y_mm)):
        raise ValueError("Axial coordinates must be finite.")
    if np.any(y_mm < 0) or np.any(y_mm > height_mm):
        raise ValueError("Axial coordinates must lie between 0 and the height.")

    centered = (y_mm - height_mm / 2) / transfer_length_mm
    half_span = height_mm / (2 * transfer_length_mm)
    magnitude = np.abs(centered)
    denominator = 1 + np.exp(-2 * half_span)
    scaled_positive = np.exp(magnitude - half_span)
    scaled_negative = np.exp(-magnitude - half_span)
    sinh_ratio = np.sign(centered) * (scaled_positive - scaled_negative) / denominator
    cosh_ratio = (scaled_positive + scaled_negative) / denominator
    displacement_mm = strain_mismatch * transfer_length_mm * sinh_ratio
    displacement_gradient = strain_mismatch * cosh_ratio
    return displacement_mm, displacement_gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height-mm", type=float, required=True)
    parser.add_argument("--transfer-length-mm", type=float, required=True)
    parser.add_argument("--strain-mismatch", type=float, required=True)
    parser.add_argument("--points", type=int, default=501)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "axial_edge_theory.csv",
    )
    args = parser.parse_args()
    if args.points < 3:
        parser.error("At least three sampling points are required.")
    coordinates = np.linspace(0, args.height_mm, args.points)
    displacement, gradient = axial_edge_solution(
        coordinates,
        args.height_mm,
        args.transfer_length_mm,
        args.strain_mismatch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["y_mm", "relative_displacement_mm", "displacement_gradient"])
        writer.writerows(zip(coordinates, displacement, gradient))
    settings = {
        "source": "Supplementary Note S6, equations S14-S17",
        "height_mm": args.height_mm,
        "transfer_length_mm": args.transfer_length_mm,
        "strain_mismatch": args.strain_mismatch,
        "points": args.points,
        "interpretation": "User-specified analytical calculation; parameters are not fitted to the measured defect profile.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
