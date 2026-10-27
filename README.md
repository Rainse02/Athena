# Athena

Athena (Analytical Tomography for Heterogeneity-aware Electrode Network Analysis)
provides the analysis code for **Mapping Curvature-Associated Structural Defects
and Heterogeneity in a Pristine Commercial Li–MnO2 Battery**.

The code analyzes three electrode defect classes: Detachment, Internal Crack,
and Surface Crack.

## Installation

```bash
pip install -r requirements.txt
```

## Analysis

All analysis scripts are in [Analysis](Analysis/).

| Script | Description |
| --- | --- |
| [Density_Thermogram.py](Analysis/Density_Thermogram.py) | Plots defect density on the flattened electrode surface. |
| [Current_Collector_Intersection_Detect.py](Analysis/Current_Collector_Intersection_Detect.py) | Detects intersections in the flattened current-collector grid. |
| [Radia_Distribution.py](Analysis/Radia_Distribution.py) | Calculates the radial distribution of defects. |
| [Vertical_Distribution.py](Analysis/Vertical_Distribution.py) | Compares axial defect profiles with the distribution of collector intersections. |
| [Angle_Distribution.py](Analysis/Angle_Distribution.py) | Calculates the angular distribution of defects. |
| [Curvature_Distibution.py](Analysis/Curvature_Distibution.py) | Fits the electrode winding curve, assigns local curvature to defect objects, and plots their volume-weighted distributions. |
| [Flatten_Intensity.py](Analysis/Flatten_Intensity.py) | Calculates defect intensity along the flattened electrode. |
| [fixed_center_statistics.py](Analysis/fixed_center_statistics.py) | Calculates object statistics, curvature enrichment, regional comparisons, and sensitivity to component volume and curvature intervals. |
| [rebuild_saved_fit.py](Analysis/rebuild_saved_fit.py) | Fits the Archimedean spiral to the extracted electrode contour. |
| [node_alignment.py](Analysis/node_alignment.py) | Calculates the period and lag correlation of axial defect and collector-node profiles. |
| [axial_ends.py](Analysis/axial_ends.py) | Summarizes defect and collector-node profiles in ten axial intervals. |
| [validation_metrics.py](Analysis/validation_metrics.py) | Calculates Dice, IoU, precision, and recall from segmentation validation counts. |

Functions for collector analysis (`collector.py`), spatial
profiles (`spatial.py`), density estimation (`density.py`), curve geometry
(`curvature_geometry.py`), path normalization (`normalization.py`), regional
statistics (`regions.py`), and volume sensitivity (`volume_sensitivity.py`)
support these calculations.

Set the input and result directories in [settings.py](Analysis/settings.py).
Image files, contour points, object measurements, and profile tables are
selected in the corresponding scripts. For example:

```bash
python Analysis/Curvature_Distibution.py
python Analysis/fixed_center_statistics.py
python Analysis/node_alignment.py
```

## Citation

Yifei Yan, Kaile Niu, Lei Cai, Zhengxiang Song, Ke Du, Guibin Zan,
Fangyang Liu, and Jinhao Meng.

**Mapping Curvature-Associated Structural Defects and Heterogeneity in a
Pristine Commercial Li–MnO2 Battery.**

Citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

[Apache License 2.0](LICENSE).
