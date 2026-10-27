# Validation scope

## Original-source completeness

The seven Python scripts in `Analisys/` were retained directly from Athena
commit `70775b1`. A Git content comparison against that revision showed no
changes to those scripts. Their numerical functions, top-level workflows,
and original comments are present in full. A separate README documents their
input paths and environment requirements.

The original scripts retain their historical local path settings and require
the corresponding images, stacks, and workbooks. Source preservation is not a
claim that all seven original workflows were rerun on their original data.

## Configured workflow

Execution checks use Python 3.12.5 and the pinned dependencies in
`requirements.txt`. Study inputs are read from an external directory, and
all generated files are written outside the repository. These checks concern
the configured entry points in `code/`, whose input/output adapters were
updated for the code-only repository.

Checks completed on 2026-10-27:

| Check | Outcome |
| --- | --- |
| Complete saved-table analysis using an external data root | Completed, exit code 0 |
| Complete mechanics post-processing using an external data root | Completed, exit code 0 |
| Flattened-image analysis using the relocated configuration | Completed, exit code 0 |
| Help for 13 executable entries without data | All returned exit code 0 |
| Missing-input behavior for the three principal entries | All returned the expected exit code 2 with a concise error and no traceback |
| Environment-variable roots for direct analysis and mechanics modules | Both completed, exit code 0 |
| Two existing axial-edge equation/boundary tests | Both passed |

The image check covered the flattened-profile method. Radial, angular, and
vertical methods were not rerun on complete segmented stacks. The original
seven scripts were preserved without reformatting. The configured modules
were formatted with Black 24.4.2.

The citation metadata were parsed as YAML and checked for required fields and
the eight recorded authors. No release or DOI was issued during these checks.

## Scientific boundary

The numerical functions used by the configured workflow had previously been
compared with the saved study summaries. Statistical tables agreed within
floating-point precision (the largest observed absolute difference in a
regional table was approximately 1.03 × 10⁻¹²); the checked mechanics summaries
reproduced their saved values. The external-path adaptation changes input and
output selection, command-line behavior, and module naming.

These checks do not recreate raw tomography, external image unfolding,
segmentation training or inference, full segmented-stack processing, or new
finite-element solutions. Input data and native models are outside this
software release.
