# Athena: battery electrode defect analysis

Athena contains the study-specific Python code accompanying **Mapping
Curvature-Associated Structural Defects and Heterogeneity in a Pristine
Commercial Li–MnO2 Battery**. The code quantifies material and structural
defects in an electrochemical battery cell after image segmentation and
processes numerical mechanics results used in the study.

The repository contains source code, analysis configuration, and documentation.
Image volumes, measured tables, numerical field exports, trained weights,
third-party framework source, and native simulation models are outside this
software repository. Running the study analyses requires the external inputs
described in [docs/INPUTS.md](docs/INPUTS.md).

## Contents

| Location | Purpose |
| --- | --- |
| [Analisys/](Analisys/README.md) | All seven complete original analysis scripts retained from the original repository |
| [code/analysis/](code/analysis/README.md) | Winding geometry, defect statistics, exposure normalization, sensitivity analysis, axial profiles, segmentation metrics, and image-analysis methods |
| [code/mechanics/](code/mechanics/README.md) | Surface/interface statistics, winding-path deformation summaries, and the axial-edge analytical solution |
| [configs/](configs/) | Image-analysis settings and input-path configuration |
| [docs/WORK_DESCRIPTION.md](docs/WORK_DESCRIPTION.md) | Scientific methods, manuscript correspondence, and interpretation limits |
| [docs/INPUTS.md](docs/INPUTS.md) | External input structure, fields, and units |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Execution checks and their scope |
| [CITATION.cff](CITATION.cff) | Machine-readable software citation metadata |
| [docs/ZENODO.md](docs/ZENODO.md) | Preparation of a versioned software archive and DOI |

## Environment

Use a dedicated Python environment. The selected analyses were checked with
Python 3.12.5 and the versions listed in `requirements.txt`.

```console
python -m pip install -r requirements.txt
```

For the original standalone scripts in `Analisys/`, install their additional
dependencies with `python -m pip install -r requirements-original.txt`. Their
README describes the original path settings and expected inputs.

NumPy, SciPy, pandas, and Matplotlib support the numerical analyses. The image
methods additionally use OpenCV with `ximgproc`, scikit-image, tifffile, and
Pillow. The scripts do not require a segmentation training environment or a
finite-element solver.

## Running the analyses

Run commands from the repository root. `--data-root` is an external directory
containing the `analysis/` and `mechanics/` input subdirectories. Outputs are
written to the directory supplied with `--output-root`.

```console
python code/analysis/run_all.py --data-root /path/to/study-data --output-root /path/to/results
python code/mechanics/run_all.py --data-root /path/to/study-data --output-root /path/to/results
```

For Windows, use a quoted path such as `"E:/research/study-data"`.
The directory structure must follow [docs/INPUTS.md](docs/INPUTS.md).
There are no measured data files in a fresh clone.

Select an individual statistical task with `--task`:

```console
python code/analysis/run_all.py --task nodes --data-root /path/to/study-data --output-root /path/to/results
python code/analysis/run_image_tools.py --help
python code/mechanics/axial_edge_theory.py --help
```

The axial-edge evaluator is independent of measured data. It requires explicit
axial height, transfer length, and mismatch strain. Run its equation and
boundary-condition checks with:

```console
python code/mechanics/test_axial_edge_theory.py
```

## Scope of reproduction

Statistical reproduction requires the recorded contour, object measurements,
regional counts, and profiles. Mechanics post-processing requires exported
numerical samples. These scripts do not regenerate raw image reconstruction,
external unfolding, segmentation, or finite-element solutions.

The three analyzed classes are Detachment, Internal Crack, and Surface Crack.
Calculations retain the study's fixed-center winding fit, coordinate
conventions, physical spacing, percentile definitions, and analysis intervals.
Review these settings before applying the code to another cell.
[WORK_DESCRIPTION.md](docs/WORK_DESCRIPTION.md) explains the assumptions and
distinguishes descriptive associations from mechanical failure criteria.

## Citation and license

Software metadata are in [CITATION.cff](CITATION.cff); manuscript information
is in [CITATION.md](CITATION.md). A DOI and release date will be added when a
release is archived. No DOI has been assigned by this preparation step.

The repository retains its Apache License 2.0 licensing; see [LICENSE](LICENSE).
Dependencies are installed separately and retain their respective licenses.
