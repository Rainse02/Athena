# Here is the Athena for lithium battery defects identification and analysis!
Athena means 
Analytical Tomography for Heterogeneity-aware Electrode Network Analysis.
## Micro Defects identification model
The identification model for electrode defects in our work is constructed based the open-source project [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

### Installation
To use nnU-Net, you should install [PyTorch](https://pytorch.org/get-started/locally/) firstly,
and then install the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). For example, you can install it like this:

    pip install nnunetv2

### Dataset structure
The structure of our training dataset is like this: 

    Dataset001_BatteryElectrode/
    ├── dataset.json
    ├── imagesTr
    └── labelsTr

And our dataset.json is:

    {
        "channel_names": {
            "0": "T2"
        },
        "labels": {
            "background": 0,
            "Detach": 1,
            "InternalCrack": 2,
            "SurfaceCrack": 3,
        },
        "numTraining": 125,
        "file_ending": ".tif"
    }

### Training and Identification
The plain model is used in this work for its stability. The detailed process is shown as following:
1)  The data process 

        nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
2) The training process:

        nnUNetv2_train 001 2d 1 --npz   
3) The identification process:

        nnUNetv2_predict -i in_path -o out_path -d 001 -c 2d -f 0 --save_probabilities
    The **in_path** is the folder of original CT images, and the **out_path** is the identification results folder.

For more detailed information about the nnU-Net, see the [readme of nnU-Net](https://github.com/MIC-DKFZ/nnUNet/blob/master/readme.md).

## Analysis
After the identification of micro electrode defects, 
we have developed the following programs for statistical analysis of the distribution characteristics of defects.

### [Density_Thermogram](Analisys%2FDensity_Thermogram.py):
This program is used to draw the distribution of defects on the surface of the electrode after it is flattened, which is realized in a professional software.

### [Current_Collector_Intersection_Detect](Analisys%2FCurrent_Collector_Intersection_Detect.py):
This program is used to detect and draw the distribution of **key grid nodes** after the current collector is flattened.

### [Radia_Distribution](Analisys%2FRadia_Distribution.py):
This program is used to statistically analyze and plot the distribution of defects in the **radial** direction of the battery.

### [Vertical_Distribution](Analisys%2FVertical_Distribution.py):
This program is used to statistically analyze and plot the distribution of defects in the **vertical** direction of the battery.

### [Angle_Distribution](Analisys%2FAngle_Distribution.py):
This program is used to statistically analyze and plot the distribution of defects in the **angle** direction of the battery.

### [Curvature_Distibution](Analisys%2FCurvature_Distibution.py):
This program is used to statistically analyze the distribution of defects in electrode with different curvatures. 
By using the winding curve of the electrode (obtained in another software), fit the standard [Archimedean spiral](https://en.wikipedia.org/wiki/Archimedean_spiral) to obtain the curvature of each position of the electrode coil.
Further associate different defect individuals with points on the curve, obtain the curvature size of each defect individual, and plot the kernel density distribution image
  
### [Flatten_statistics](Analisys/Flatten_Intensity.py):
This program is used to statistically analyze the defect distribution on the flatten electrode. From one side to another side along with the flatten direction.


## License
Athena is open source. For more information about its license, see [LICENSE](LICENSE).

## Citation
    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.