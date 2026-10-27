# Study methods and code correspondence

The study examines material and structural defects in one pristine commercial
Li–MnO2 electrochemical cell using reconstructed X-ray computed-tomography
images. The software starts from segmented images or extracted measurements.
Its outputs describe defect distributions relative to the wound electrode
geometry and collector structure.

All seven original standalone analysis scripts are preserved in `Analisys/`.
Their original algorithms and helper functions remain complete. The configured
modules in `code/analysis/` supply the manuscript's selected statistical
workflow and subsequent calculations; they do not replace the archived
original implementations.

## Statistical analysis

The analyzed classes are Detachment, Internal Crack, and Surface Crack.
Connected-component measurements supply object volume and centroid position.
The transverse electrode path is represented by an Archimedean spiral fitted
to extracted contour points with the image center fixed at `(613, 603.5)`
pixels. The fit uses `r = a + b*theta`, with `a = 427.3311730021125` pixels and
`b = 6.561336764513805` pixels in the selected study fit. The isotropic voxel
spacing is `0.01684976915031592 mm`.

Defects receive curvature through their nearest extracted-contour point. The
fitted ideal path supplies the exposure denominator: the fraction of path
length in each curvature interval. Volume enrichment is the fraction of a
class's defect volume in an interval divided by the path-length fraction in
that interval. Object-count enrichment is calculated analogously. The main
intervals are below 0.2, from 0.2 to below 0.3, and at least 0.3 mm⁻¹.

Regional comparisons retain matched spatial windows and exposure denominators.
Minimum-volume filtering and neighboring interval boundaries quantify
sensitivity to component resolution and interval selection. They do not
introduce additional cells or independent replicates.

The axial analysis uses paired 1,590-position Detachment and collector-node
profiles. It calculates raw and background-subtracted Pearson correlations
with nonwrapping overlap over lags from −37 to +37 pixels. Background removal
uses the retained 151-position running mean. Positive lag moves the Detachment
profile toward the cell top. Ten equal-height intervals describe broad axial
variation separately from local node correspondence.

Validation recalculates mean per-case Dice and IoU and pooled precision and
recall from saved measurements. It does not perform new segmentation inference.
Image-analysis functions also support collector intersection detection and
flattened, radial, angular, and axial distributions from supplied images.

## Mechanics calculations

The mechanics modules read numerical field exports to calculate surface
curvature, signed tensile-stress statistics, interface-traction statistics,
and accumulated winding-path deformation. Geometry definitions and numerical
samples are external inputs. Post-processing preserves the study's sample
domains, stress normalization, and percentile conventions. The original
matrix summaries use nearest-rank P95, the final signed axial-stress summary
uses NumPy's linear percentile interpolation, and interface P95 uses arc-length
weights.

The separate axial-edge evaluator implements the bonded-layer solution in
Note S6. With axial height `H`, transfer length `ell`, and mismatch strain
`Delta_epsilon`, the relative displacement is

```text
delta(y) = Delta_epsilon * ell * sinh((y - H/2)/ell) / cosh(H/(2*ell)).
```

It satisfies `delta'' = delta/ell²` and
`delta'(0) = delta'(H) = Delta_epsilon`. Inputs must be supplied explicitly.
The winding-path transfer length is not an estimate of the axial-edge transfer
length. This evaluator directly implements the manuscript equations; it does
not fit the measured axial defect profile.

## Connection to the manuscript

| Manuscript component | Code |
| --- | --- |
| Object inventory; Table S2 | `code/analysis/fixed_center_statistics.py` |
| Winding geometry; Figure S11, Table S5 | `rebuild_saved_fit.py`, `curvature_geometry.py`, `fixed_center_statistics.py` |
| Curvature enrichment; Figure 6C, Table S6 | `normalization.py`, `fixed_center_statistics.py` |
| Regional comparisons; Table S7 | `regions.py` |
| Volume and interval sensitivity; Figure S15, Tables S8/S12 | `volume_sensitivity.py`, `regions.py` |
| Validation metrics; Table S4 | `validation_metrics.py` |
| Axial/node profiles; Note S6, Figure S16 | `node_alignment.py`, `axial_ends.py` |
| Axial-edge analytical solution; Note S6 | `code/mechanics/axial_edge_theory.py` |
| Local bending; Figure 6E, Figure S18, Tables S3/S10 | `analyze_bending.py` |
| Local mismatch statistics; Figure 6F, Tables S9/S10 | `analyze_end_bending.py` |
| Accumulated winding-path deformation; Figure S17, Table S11 | `analyze_cumulative_transfer.py` |
| Symmetric strain and stress normalization; Note S8 | `analyze_surface_strain.py` |

Filenames without paths refer to `code/analysis/` in statistical rows and
`code/mechanics/` in mechanics rows. Provenance is recorded in
`code/analysis/SOURCE_MAP.json` and `code/mechanics/source_provenance.json`.

## Interpretation and reuse

Defect–curvature associations are descriptive within-cell results. Elastic
stress calculations concern specified loads and material assumptions; their
values are not calibrated fracture thresholds. Applying the code to another
cell requires consistent coordinates, physical spacing, contour extraction,
object measurements, and analysis domains. Reproducing the paper's numerical
values requires its corresponding measured inputs as well as the software.
