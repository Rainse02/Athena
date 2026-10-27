# Athena for battery defect identification and analysis

Athena (Analytical Tomography for Heterogeneity-aware Electrode Network Analysis)
provides code for identifying and analyzing structural defects in battery electrodes.

The code analyzes three electrode defect classes: Detachment, Internal Crack,
and Surface Crack.

## Defect identification

Electrode defects are identified using [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

### Installation

Install [PyTorch](https://pytorch.org/get-started/locally/), followed by nnU-Net:

```bash
pip install nnunetv2
```

Install the analysis dependencies:

```bash
pip install -r requirements.txt
```

### Dataset structure

The training dataset is organized as follows:

```text
Dataset001_BatteryElectrode/
├── dataset.json
├── imagesTr/
│   └── Battery_0001_0000.tif
└── labelsTr/
    └── Battery_0001.tif
```

The `_0000` suffix identifies the image channel. Image and label filenames
share the same case identifier. The [dataset.json](Segmentation/dataset.json)
configuration is:

```json
{
  "channel_names": {
    "0": "T2"
  },
  "labels": {
    "background": 0,
    "Detach": 1,
    "InternalCrack": 2,
    "SurfaceCrack": 3
  },
  "numTraining": 125,
  "file_ending": ".tif"
}
```

`T2` is the channel identifier used in the project configuration; the images
are battery CT slices.

Set the nnU-Net working directories and place `Dataset001_BatteryElectrode`
under `nnUNet_raw`:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### Training and identification

The project uses the 2D plain U-Net configuration with dataset ID `001`
and fold `1`.

1. Plan and preprocess the dataset:

   ```bash
   nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
   ```

2. Train the model:

   ```bash
   nnUNetv2_train 001 2d 1 --npz
   ```

3. Predict defect labels:

   ```bash
   nnUNetv2_predict -i in_path -o out_path -d 001 -c 2d -f 1 --save_probabilities
   ```

`in_path` contains the CT images for prediction, named with the `_0000.tif`
channel suffix. `out_path` receives the predicted labels and probabilities.

These commands are also provided in [preprocess.sh](Segmentation/preprocess.sh),
[train.sh](Segmentation/train.sh), and [predict.sh](Segmentation/predict.sh).
For example, `bash Segmentation/predict.sh in_path out_path` runs prediction.

Further usage details are available in the
[nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md).

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

## License

[Apache License 2.0](LICENSE).
