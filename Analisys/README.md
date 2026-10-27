# Complete original analysis scripts

This directory retains all seven original Athena analysis scripts. Their
contents are unchanged from the source files in repository commit `70775b1`.
The original directory and filenames are retained, including their spelling.
The scripts are complete implementations, not forwarding wrappers or selected
function extracts.

| Original script | Input and implemented analysis |
| --- | --- |
| `Angle_Distribution.py` | Internal Crack TIFF slices; accumulation of the first 500 slices and one-degree angular intensity in the original annular domain |
| `Current_Collector_Intersection_Detect.py` | Flattened collector image; resizing, thinning, convolution, thresholding, morphology, contour filtering, and intersection-map output |
| `Curvature_Distibution.py` | Contour image and three object-measurement workbooks; contour extraction, fixed-center spiral fitting, curvature assignment, nearest-contour mapping, and weighted density plots |
| `Density_Thermogram.py` | Flattened defect image; convolution density and Gaussian, distance-transform, sliding-window, kernel-density, morphological, and image-coloring functions |
| `Flatten_Intensity.py` | Flattened surface-defect image; thresholded column sums, physical-distance conversion, and workbook output |
| `Radia_Distribution.py` | Internal Crack TIFF slices; accumulation, radial bins, the original radius-squared scaling, normalization, and plotting |
| `Vertical_Distribution.py` | Detachment TIFF slices and collector intersection image; axial sums, normalized profiles, gradient-band visualization, and overlaid plotting |

## Environment and execution

Install the original scripts' dependencies from the repository root:

```console
python -m pip install -r requirements-original.txt
```

The original scripts execute at top level and use path variables embedded in
the source. Set their input paths and output destinations to existing local
locations before running them. They also retain the original image dimensions,
spatial windows, plotting behavior, and workbook assumptions. In particular,
`Flatten_Intensity.py` appends a worksheet to an existing workbook. A fresh
clone contains no images, segmentation stacks, or object workbooks.

The original curvature script expects object workbooks named `Internal255.xlsx`,
`Detach255.xlsx`, and `Surface255.xlsx`. It drops the first data row, reads
volume from the first column, and transverse centroid coordinates from the
third and fourth columns. These are original input conventions, not a general
spreadsheet importer.

## Relation to the configured analysis code

The modules under `code/analysis/` provide configurable entry points and the
additional normalization, region, sensitivity, validation, and axial analyses
used in the manuscript. Both sets are retained: the original scripts document
the full initial implementations, and the configured workflow documents its
specific statistical definitions. Original display normalizations and later
path- or area-exposure statistics need not be numerically identical.

Original-source completeness was checked against the Git revision above.
The configured workflow was run with external study inputs; this does not
mean that every original top-level script was rerun on its original full
image stack. See [validation scope](../docs/VALIDATION.md).
