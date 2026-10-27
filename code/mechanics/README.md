# Mechanics numerical post-processing

These scripts summarize external numerical exports from the study's mechanics
calculations. This repository contains post-processing and analytical source
code; models, input measurements, exports and reference results are supplied
separately. The scripts do not start COMSOL.

Run from the repository root after installing `requirements.txt`:

```console
python code/mechanics/run_all.py --data-root /path/to/data --output-root /path/to/results
```

`--data-root` is the parent containing `analysis/` and `mechanics/`.
`--output-root` is the output parent; results are written beneath its
`mechanics/` subdirectory. `ATHENA_DATA_ROOT` and `ATHENA_OUTPUT_ROOT`
provide defaults for the options and for direct module execution. Otherwise
roots default to repository `data/` and `outputs/`; inputs must be obtained
separately. Relative command line paths use the current working directory.
The driver sets roots before importing numerical modules.

For direct execution in PowerShell:

```powershell
$env:ATHENA_DATA_ROOT = 'E:\external\study\data'
$env:ATHENA_OUTPUT_ROOT = 'E:\external\study\results'
python code/mechanics/analyze_bending.py
```

Executable entries support `--help` without data. Missing files produce a
concise message identifying the required input. See
[the input contract](../../docs/INPUTS.md) for filenames, columns and units.

## Calculations

| Entry | External input | Output beneath `mechanics/` and paper correspondence |
| --- | --- | --- |
| `analyze_bending.py` | Final offset/reference surface and force-balance tables | `bending_summary.json`: signed surface stress P95, circular surface fits and end-force balance; Figure 6E/S18 and Table S10 |
| `analyze_end_bending.py` | Base, zero-increment replay, constrained mismatch and normal-free exports; geometry regions | `end_bending_summary.json` and `current_axial_statistics.json`: matrix/interface P95, surface curvature, displacement increments and compatibility; Figure 6F, Note S9 and Table S10 |
| `analyze_cumulative_transfer.py` | One-, three-, six-turn and full-winding finite-end profiles; fit metadata | `cumulative_transfer_summary.csv` and JSON: independent winding-prefix summaries; Figure S17 and Table S11 |
| `analyze_surface_strain.py` | Published thickness and normalization parameters | `surface_strain_estimate.csv` and `stress_normalization.json`: symmetric estimate and stress references in Note S8 |
| `axial_edge_theory.py` | Explicit electrode axial height, transfer length and mismatch strain | Separate user-specified analytical evaluation from Note S6 |

The original percentile, polygon-distance, interval-mask, surface-fit and
profile-summary algorithms are retained. `source_provenance.json` records
original module names and the path/entry adaptations.

Surface bending P95 uses the original central interval. Matrix mismatch
statistics sample cathode points near the collector using externally supplied
node and adjacent ordinary intervals. The end-bending matrix summary retains
the nearest-rank percentile; the current axial statistics retain NumPy's
linear-interpolation P95. Interface percentiles use arc-length weights.
Vertical artificial end edges are omitted from the point-to-interface
distance calculation. The mismatch surface fit is quadratic; the bending
comparison uses a circular fit.

Winding profiles represent independently equilibrated prefixes. Their
finite-end force uses the Robin boundary relation; the discrepancy with the
sampled endpoint gradient is retained as a diagnostic. The positional reader
assigns `N_over_Aeq` to the dimensionless force proxy, including when a
historical header adds an erroneous `_per_mm` suffix.

## Separate axial-edge evaluator

```console
python code/mechanics/axial_edge_theory.py --height-mm 20 --transfer-length-mm 2 --strain-mismatch 0.001 --output /path/to/results/axial_edge.csv
```

The example physical inputs illustrate the CLI. The axial-edge coordinate
spans electrode height. Transfer length and mismatch strain are specified by
the caller and are not fitted to the CT profiles. The separate winding
model's transfer length is not adopted for this evaluator. If `--output` is
omitted, its CSV and settings JSON are written beneath the mechanics output
root.

The symmetric surface-strain estimate uses `epsilon = kappa*h/2`, assuming
zero membrane strain and a mid-plane neutral axis in a symmetric section.
Geometry and normalization constants come from the external parameter JSON.
Saved-export runs check numerical post-processing; they do not validate a new
mechanics solve, rebuild geometry or repeat mesh studies.
