"""Summarize the four finite-end winding prefixes adopted in Figure S17."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from package_paths import DATA_ROOT, OUTPUT_ROOT, run_cli


DATA_DIR = DATA_ROOT
INPUT_DIR = DATA_DIR / "cumulative_transfer"
OUTPUT_DIR = OUTPUT_ROOT
D_MM = 29.0 * 0.016849769150315919 / 6.0
PROFILE_COLUMNS = [
    "X",
    "s_mm",
    "theta_rad",
    "r_mm",
    "Psi",
    "Psi_comsol",
    "U",
    "U_minus_Psi",
    "U_X",
    "G_mm",
    "v_mm",
    "slip_mm",
    "N_over_Aeq",
    "tau_over_Aeq_per_mm",
    "beta",
]
PROFILE_RE = re.compile(
    r"^(?P<stage>stage_(?:1turn|3turn|6turn)|full)_"
    r"(?P<bc>finite_end_restraint)_ell_(?P<ell>5)mm"
    r"(?P<mesh>_mesh8192)?_profile\.txt$"
)
STAGES = ("stage_1turn", "stage_3turn", "stage_6turn", "full")


def read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    values = np.loadtxt(path, comments="%", dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.size == 0:
        raise ValueError(f"Empty COMSOL table: {path.name}")
    if values.shape[1] != len(columns):
        raise ValueError(
            f"Unexpected column count in {path.name}: "
            f"{values.shape[1]} != {len(columns)}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite data in COMSOL table: {path.name}")
    return pd.DataFrame(values, columns=columns)


def parse_profile(path: Path) -> tuple[dict, pd.DataFrame]:
    match = PROFILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected profile filename: {path.name}")
    metadata = match.groupdict()
    metadata["ell_mm"] = float(metadata["ell"].replace("p", "."))
    metadata["mesh_elements"] = 8192 if metadata["mesh"] else 4096
    metadata["is_refinement"] = bool(metadata["mesh"])
    table = read_table(path, PROFILE_COLUMNS)
    return metadata, table


def summarize_profile(metadata: dict, table: pd.DataFrame) -> dict:
    ell_mm = metadata["ell_mm"]
    span_mm = float(table.s_mm.iloc[-1])
    slip_mm = D_MM * table.U_minus_Psi.to_numpy()
    force_norm = table.U_X.to_numpy()
    tau_norm = table.U_minus_Psi.to_numpy()
    n_over_aeq = D_MM / span_mm * force_norm
    tau_over_aeq = D_MM / (ell_mm * ell_mm) * tau_norm
    abs_slip = np.abs(slip_mm)
    peak_index = int(np.argmax(abs_slip))
    psi_end = float(table.Psi.iloc[-1])
    summary = {
        "stage": metadata["stage"],
        "boundary_condition": metadata["bc"],
        "ell_mm": ell_mm,
        "mesh_elements": metadata["mesh_elements"],
        "sample_count": int(len(table)),
        "span_S_mm": span_mm,
        "theta_start_rad": float(table.theta_rad.iloc[0]),
        "theta_end_rad": float(table.theta_rad.iloc[-1]),
        "turns": (float(table.theta_rad.iloc[-1]) - float(table.theta_rad.iloc[0]))
        / (2.0 * np.pi),
        "r_start_mm": float(table.r_mm.iloc[0]),
        "r_end_mm": float(table.r_mm.iloc[-1]),
        "Psi_end": psi_end,
        "G_end_mm": float(table.G_mm.iloc[-1]),
        "v_end_mm": float(table.v_mm.iloc[-1]),
        "end_slip_mm": float(slip_mm[-1]),
        "max_abs_slip_mm": float(abs_slip.max()),
        "p95_abs_slip_mm": float(np.quantile(abs_slip, 0.95)),
        "peak_abs_slip_s_mm": float(table.s_mm.iloc[peak_index]),
        "peak_abs_slip_r_mm": float(table.r_mm.iloc[peak_index]),
        "max_abs_N_over_Aeq": float(np.abs(n_over_aeq).max()),
        "p95_abs_N_over_Aeq": float(np.quantile(np.abs(n_over_aeq), 0.95)),
        "max_abs_tau_over_Aeq_per_mm": float(np.abs(tau_over_aeq).max()),
        "p95_abs_tau_over_Aeq_per_mm": float(np.quantile(np.abs(tau_over_aeq), 0.95)),
        "max_abs_N_normalized": float(np.abs(force_norm).max()),
        "p95_abs_N_normalized": float(np.quantile(np.abs(force_norm), 0.95)),
        "max_abs_tau_normalized": float(np.abs(tau_norm).max()),
        "p95_abs_tau_normalized": float(np.quantile(np.abs(tau_norm), 0.95)),
        "max_abs_Psi_table_consistency": float(
            np.abs(table.Psi.to_numpy() - table.Psi_comsol.to_numpy()).max()
        ),
        "derived_force_max_abs_difference": float(
            np.abs(n_over_aeq - table.N_over_Aeq.to_numpy()).max()
        ),
        "derived_tau_max_abs_difference": float(
            np.abs(tau_over_aeq - table.tau_over_Aeq_per_mm.to_numpy()).max()
        ),
        "geometry_scope": "continuous original Archimedean-spiral prefix",
        "stage_interpretation": "independently equilibrated prefix from theta0; no irreversible winding time history",
        "node_pitch_note": "75 px is an axial image-node pitch and is not used as an along-winding s pitch",
        "model_family": "normalized 1D coefficient-form PDE",
        "tau_definition": "interface transfer traction proxy tau/Aeq=d/ell^2*(U-Psi), distinct from bulk 2D shear",
    }
    raw_endpoint_force = float(n_over_aeq[-1])
    summary["raw_endpoint_N_over_Aeq"] = raw_endpoint_force
    if metadata["bc"] == "free":
        summary["boundary_consistent_endpoint_N_over_Aeq"] = 0.0
        summary["endpoint_force_discrepancy"] = raw_endpoint_force
        summary["endpoint_force_note"] = (
            "natural zero-flux endpoint; sampled linear-FE gradient is a diagnostic"
        )
    elif metadata["bc"] == "finite_end_restraint":
        gamma = 0.001
        boundary_force = -gamma * float(table.v_mm.iloc[-1])
        summary["boundary_consistent_endpoint_N_over_Aeq"] = boundary_force
        summary["endpoint_force_discrepancy"] = raw_endpoint_force - boundary_force
        summary["endpoint_force_note"] = (
            "Robin endpoint force; sampled linear-FE gradient discrepancy is a diagnostic"
        )
    else:
        summary["boundary_consistent_endpoint_N_over_Aeq"] = None
        summary["endpoint_force_discrepancy"] = None
        summary["endpoint_force_note"] = (
            "locked endpoint has derivative-bound response only; large values are not a physical force estimate"
        )
    if metadata["bc"] == "finite_end_restraint":
        gamma = 0.001
        rho = gamma * span_mm
        robin_residual = table.U_X.to_numpy()[-1] + rho * table.U.to_numpy()[-1]
        summary["finite_end_gamma_per_mm"] = gamma
        summary["finite_end_rho_gamma_S"] = rho
        summary["robin_endpoint_residual"] = float(robin_residual)
    return summary


def main() -> None:
    records = []
    for stage in STAGES:
        path = INPUT_DIR / f"{stage}_finite_end_restraint_ell_5mm_profile.txt"
        metadata, table = parse_profile(path)
        records.append(summarize_profile(metadata, table))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / "cumulative_transfer_summary.csv", index=False
    )
    fit = json.loads((DATA_DIR / "curvature_summary.json").read_text(encoding="utf-8"))[
        "fit"
    ]
    payload = {
        "model_family": "normalized 1D coefficient-form PDE",
        "governing_equation": "-U_XX + beta^2*(U-Psi)=0",
        "relative_extension_definition": "v=uc-um; delta=v-G",
        "effective_path_separation_mm": D_MM,
        "transfer_length_mm": 5.0,
        "outer_restraint_coefficient_per_mm": 0.001,
        "mesh_elements": 4096,
        "geometry_fit": fit,
        "stage_interpretation": "independently equilibrated prefixes from a common starting position",
        "canonical_units": {
            "X": "dimensionless coordinate s/S",
            "G_mm": "geometric path-length difference in mm",
            "v_mm": "differential extension in mm",
            "slip_mm": "v-G in mm",
            "N_over_Aeq": "dimensionless force proxy; legacy TXT header may say N_over_Aeq_per_mm",
            "tau_over_Aeq_per_mm": "interface traction proxy in 1/mm",
        },
        "cases": records,
    }
    (OUTPUT_DIR / "cumulative_transfer_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                record["stage"]: {
                    "S_mm": record["span_S_mm"],
                    "G_end_mm": record["G_end_mm"],
                    "v_end_mm": record["v_end_mm"],
                    "slip_end_mm": record["end_slip_mm"],
                }
                for record in records
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_cli(main, __doc__)
