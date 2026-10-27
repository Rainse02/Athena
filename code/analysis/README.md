# Analysis code

These drivers reproduce the study's selected post-segmentation calculations
for Detachment, Internal Crack and Surface Crack. Inputs are external saved
contour points, object tables, angular counts, axial profiles and validation
counts. This repository contains source code and method settings; input data
and reference results are supplied separately. The complete original analysis
scripts are retained separately under `Analisys/`; these adapted drivers
provide the study-selected calculations with an external input contract.

Run from the repository root after installing `requirements.txt`:

```console
python code/analysis/run_all.py --data-root /path/to/data --output-root /path/to/results
python code/analysis/run_all.py --task fixed_center --data-root /path/to/data --output-root /path/to/results
python code/analysis/run_all.py --task nodes --data-root /path/to/data --output-root /path/to/results
python code/analysis/run_all.py --task axial_ends --data-root /path/to/data --output-root /path/to/results
python code/analysis/run_all.py --task validation --data-root /path/to/data --output-root /path/to/results
```

`--data-root` is the parent containing `analysis/` and `mechanics/`.
`--output-root` is the output parent; this entry writes beneath its `analysis/`
subdirectory. `ATHENA_DATA_ROOT` and `ATHENA_OUTPUT_ROOT` provide defaults for
the options and for direct module execution. If neither is set, roots default
to repository `data/` and `outputs/`; input files must be obtained separately.
Relative command line paths use the current working directory. The driver
sets roots before importing numerical modules.

For direct execution in PowerShell:

```powershell
$env:ATHENA_DATA_ROOT = 'E:\external\study\data'
$env:ATHENA_OUTPUT_ROOT = 'E:\external\study\results'
python code/analysis/node_alignment.py
```

See [the input contract](../../docs/INPUTS.md) for filenames, schemas,
coordinates and units. Executable entries support `--help` without input data.
Missing files produce an actionable error identifying the external location.

## Calculations

| Entry | External input | Output beneath `analysis/` |
| --- | --- | --- |
| `fixed_center_statistics.py` | Contour-order points, ideal path, nearest-contour assignments, three object tables and cached angular measurements | `fixed_center/`: curvature mappings/bins, exposure normalization, radial/angular regions, thickness and volume-cutoff sensitivity |
| `rebuild_saved_fit.py` | Extracted contour points in recorded traversal order and fit metadata | `fixed_center/saved_fit_reconstruction.json`: original two-parameter fixed-center fit and comparison with recorded parameters |
| `node_alignment.py` | Common 1,590-row Detachment and node profiles | `nodes/`: period estimates, nonwrapping lag scans, zero-lag and +12-pixel correlations |
| `axial_ends.py` | The same common axial profiles | `axial_ends/`: ten equal-height bins and lower/central/upper descriptive comparisons |
| `validation_metrics.py` | Three-class per-case TP/FP/FN/TN counts and stored Dice/IoU | `validation/`: mean per-case Dice/IoU and pooled precision/recall |

The fixed-center driver retains the original fit constants, nearest-contour
mapping and ideal-spiral path exposure calculation. It reuses angular voxel
counts extracted upstream. `legacy_` remains a historical column prefix for
the fixed-center mapping used in the study. Segmentation, unfolding and
contour extraction are upstream operations.

Dice/IoU are means of per-case values, excluding undefined metrics.
Precision/recall pool counts across those same validation cases. Profiles and
parameter sensitivity describe variation within the measured battery cell.

## Image-analysis methods

`athena_analysis/` retains the effective collector, vertical, radial, angular,
flattened-profile and density methods used by the adapted image runner:

```console
python code/analysis/run_image_tools.py collector --data-root /path/to/data --output-root /path/to/results
python code/analysis/run_image_tools.py flatten --data-root /path/to/data --output-root /path/to/results
python code/analysis/run_image_tools.py density --data-root /path/to/data --output-root /path/to/results
python code/analysis/run_image_tools.py radial --data-root /path/to/data --input /path/to/InternalCrack --output-root /path/to/results
python code/analysis/run_image_tools.py angle --data-root /path/to/data --input /path/to/InternalCrack --output-root /path/to/results
python code/analysis/run_image_tools.py vertical --data-root /path/to/data --input /path/to/Detach --collector /path/to/collector.png --output-root /path/to/results
```

The default [image configuration](../../configs/image_analysis.json) preserves
the original spacing, thresholds, bin widths, exposure definitions and seeds.
`--config` selects another configuration. Relative `inputs` paths are resolved
beneath `--data-root`, independently of the configuration's own location.
Absolute input paths are accepted. `--input` overrides the primary image or
stack, and `--collector` overrides the vertical reference. `--output-dir`
overrides the complete task directory; otherwise output is
`analysis/images/<task>/` beneath the selected output root.

All images are external. Collector uses a flattened collector TIFF and optional
reference mask; flatten/density use unfolded maps; radial/angle use ordered
Internal Crack TIFF slices; vertical uses Detachment TIFF slices and a
collector reference. Configured exposure images can replace the method's
geometric exposure definition where supported. The adapted runner does not
reconstruct upstream unfolding or the publication layout. The separate
original scripts include their original processing and plotting workflows.

`SOURCE_MAP.json` records original module names and repository adaptations.
Path selection, CLI behavior and metadata formatting are adapted for external
data; selected numerical functions are retained.

## Dependencies

The adapted analysis uses NumPy, SciPy, pandas, Matplotlib, OpenCV with
`ximgproc`, scikit-image and tifffile. Saved-table calculations use external
processed measurements and require no training framework, segmentation
weights or COMSOL solver.
