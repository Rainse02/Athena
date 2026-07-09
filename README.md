# Battery Electrode Defect Identification and Analysis

This repository provides supporting code for CT-based electrode defect identification and spatial analysis in lithium-based batteries. The segmentation step uses nnU-Net v2 as an external open-source framework. The main purpose of this repository is to support material and structural analysis after defect segmentation, rather than to introduce a new segmentation algorithm.

## Scope

This repository includes scripts for:

- preparing CT images and labels in the nnU-Net v2 dataset format;
- training or applying an nnU-Net-based segmentation model for electrode defects;
- analyzing the spatial distribution of segmented defects in flattened, radial, vertical, angular, and curvature-related coordinates.

The defect classes used in this work are:

- `Detach`: loss of contact between the current collector and active material;
- `InternalCrack`: cracks inside the electrode material;
- `SurfaceCrack`: open cracks located at the electrode surface.

## Segmentation framework

The defect segmentation workflow is based on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet). Please install and configure nnU-Net according to the official documentation.

Install PyTorch first according to your hardware environment, then install nnU-Net v2:

```bash
pip install nnunetv2
```

nnU-Net also requires the raw-data, preprocessed-data, and result directories to be defined through its environment variables. See the official [installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) and [documentation](https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation) for details.

## Dataset structure

The training dataset follows the nnU-Net v2 format:

```text
Dataset001_BatteryElectrode/
├── dataset.json
├── imagesTr/
└── labelsTr/
```

An example `dataset.json` is shown below:

```json
{
  "channel_names": {
    "0": "CT"
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

The channel name is descriptive and can be changed to match the actual CT dataset. Keep the same label definitions when reproducing a trained model.

## Training and prediction

The 2D nnU-Net configuration was used in this work for slice-based CT defect segmentation.

Preprocess the dataset:

```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

Train the model:

```bash
nnUNetv2_train 001 2d <fold> --npz
```

Run prediction:

```bash
nnUNetv2_predict -i <input_ct_folder> -o <output_folder> -d 001 -c 2d -f <fold> --save_probabilities
```

`<input_ct_folder>` is the folder containing CT images for prediction, and `<output_folder>` is the folder used to save the segmentation results. The `<fold>` used for prediction should match an available trained fold.

## Analysis scripts

After segmentation, the following scripts are used for statistical analysis of defect distributions.

| Script | Purpose |
| --- | --- |
| [`Analisys/Density_Thermogram.py`](Analisys/Density_Thermogram.py) | Plots defect density maps on the flattened electrode surface. |
| [`Analisys/Current_Collector_Intersection_Detect.py`](Analisys/Current_Collector_Intersection_Detect.py) | Detects and plots key grid nodes after flattening the current collector. |
| [`Analisys/Radia_Distribution.py`](Analisys/Radia_Distribution.py) | Analyzes defect distribution along the radial direction of the cell. |
| [`Analisys/Vertical_Distribution.py`](Analisys/Vertical_Distribution.py) | Analyzes defect distribution along the vertical direction of the cell. |
| [`Analisys/Angle_Distribution.py`](Analisys/Angle_Distribution.py) | Analyzes defect distribution along the angular direction of the cell. |
| [`Analisys/Curvature_Distibution.py`](Analisys/Curvature_Distibution.py) | Relates individual defects to local electrode curvature obtained from the fitted winding curve. |
| [`Analisys/Flatten_Intensity.py`](Analisys/Flatten_Intensity.py) | Analyzes defect distribution along the flattened electrode direction. |

Some analysis steps require external geometric inputs, such as the flattened electrode surface or the fitted winding curve. These inputs should be generated before running the corresponding scripts.

## License

This repository is released under the license provided in [`LICENSE`](LICENSE). nnU-Net is a separate upstream project and is governed by its own license.

## Citation

If the segmentation workflow based on nnU-Net is used, please cite the original nnU-Net paper:

```text
Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature Methods, 18, 203–211.
https://doi.org/10.1038/s41592-020-01008-z
```

Please also cite the related battery defect analysis paper when it becomes available.
