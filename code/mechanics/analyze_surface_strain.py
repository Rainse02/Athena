"""Evaluate the symmetric surface-strain estimate used in SI Note S8."""

from __future__ import annotations

import csv
import json
import math
import numpy as np

from package_paths import DATA_ROOT, OUTPUT_ROOT, run_cli


OUTPUT_DIR = OUTPUT_ROOT


def _validate_positive(value: float, name: str) -> float:
    """Return a finite positive scalar or raise a useful error."""

    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def symmetric_surface_strain(
    h: float,
    kappa: float | np.ndarray,
) -> float | np.ndarray:
    """Return the magnitude-signed surface strain for a symmetric section.

    Parameters
    ----------
    h:
        Section thickness in mm.
    kappa:
        Curvature in mm^-1.  The sign is retained.

    Returns
    -------
    float
        ``kappa * h / 2`` (dimensionless strain).  This expression assumes
        zero membrane strain and a neutral axis at the geometric mid-plane.
    """

    thickness_mm = _validate_positive(h, "h")
    curvature_array = np.asarray(kappa, dtype=float)
    if not np.all(np.isfinite(curvature_array)):
        raise ValueError("kappa must be finite")
    result = curvature_array * thickness_mm / 2.0
    if result.ndim == 0:
        return float(result)
    return result


def main() -> None:
    parameters = json.loads(
        (DATA_ROOT / "published_parameters.json").read_text(encoding="utf-8")
    )
    thickness_mm = parameters["local_domain_mm"][1]
    rows = []
    for curvature in (0.2, 0.3):
        strain = symmetric_surface_strain(thickness_mm, curvature)
        rows.append(
            {
                "thickness_mm": thickness_mm,
                "curvature_per_mm": curvature,
                "surface_strain_magnitude": strain,
                "surface_strain_percent": strain * 100.0,
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "surface_strain_estimate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    normalization = {
        "assumption": "symmetric rectangular-section screening, zero membrane strain, mid-plane neutral axis",
        "bending_reference_MPa": parameters["cathode_E_MPa"]
        * parameters["nominal_bending_curvature_per_mm"]
        * thickness_mm
        / 2.0,
        "mismatch_reference_MPa": parameters["cathode_E_MPa"] * parameters["F_gamma0"],
        "scope": "Kinematic screening and published normalization; not a fracture threshold or a replacement for the solved heterogeneous section",
    }
    (OUTPUT_DIR / "stress_normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"surface_strain": rows, "normalization": normalization}, indent=2)
    )


if __name__ == "__main__":
    run_cli(main, __doc__)
