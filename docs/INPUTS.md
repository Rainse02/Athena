# External input contract

Athena contains source code, method settings and work descriptions. All
measured tables, images, validation counts, mechanics exports and geometry
metadata are external inputs. Runs read these files without changing them and
write new files beneath the selected output root. This contract covers the
adapted drivers in `code/analysis/` and `code/mechanics/`. The complete
original scripts retained separately in `Analisys/` keep their original
file-selection and processing workflows.

## Selecting the roots

```console
python code/analysis/run_all.py --data-root /external/data --output-root /external/results
python code/mechanics/run_all.py --data-root /external/data --output-root /external/results
```

The data root is the parent of `analysis/` and `mechanics/`; do not pass either
subdirectory as the root. Drivers write to the corresponding subdirectories
beneath the output root. `ATHENA_DATA_ROOT` and `ATHENA_OUTPUT_ROOT` supply
defaults, including for individual numerical modules. Command line root
options take precedence. Otherwise roots default to repository `data/` and
`outputs/`. Input files are obtained separately. Relative command line paths
use the working directory. Set environment variables before starting a direct
module because it binds paths at import time.

An external study data package can be used by passing its `data` directory;
there is no need to copy it into Athena. The methods retain study-specific
constants and identity/order checks. This is a reproduction input contract,
not an automatic recalibration interface for unrelated battery cells.

## Saved-table analysis files

The full analysis requires the following paths beneath the data root.
`nodes` and `axial_ends` need only the axial CSV; `validation` needs only its
JSON; `fixed_center` needs the curvature, object and region inputs.

```text
analysis/
  curvature/
    contour_fit_points.csv
    contour_object_mapping.csv
    fit_source_metadata.json
    ideal_curve.csv
  objects/
    Detach_object_mapping.csv
    Internal_object_mapping.csv
    Surface_object_mapping.csv
    source_geometry.json
  regions/
    angle_region_comparison.csv
    angle_scope_summary.csv
    angle_5deg_histogram.csv
    angle_slice_counts.csv
  axial/
    axial_profiles.csv
  validation/
    per_case_counts.json
```

CSVs have a header row. Column names and case are significant. Preserve
recorded object identities and contour traversal order. Additional source
columns may be carried through to outputs.

## Coordinates and common units

The retained isotropic spacing is `s = 0.01684976915031592 mm/pixel`
(also mm/voxel). `Volume3d` and `volume_voxel` are voxel counts;
`volume_mm3 = volume_voxel*s^3`.

Raw centroids use image indices: `BaryCenterX=u` is the rightward column,
`BaryCenterY=v` is the downward row, and `BaryCenterZ=w` increases upward
along axial slices. In the study's 1,207-row reconstruction the paper
coordinates are `(X,Y,Z)=(u,w,1206-v)`, multiplied by `s` for mm. Thus raw
`BaryCenterY` is transverse. Fit points and object mapping use the same
`(u,v)` image frame. Angles are radians unless a field ends in `_deg`;
curvatures are inverse mm or inverse pixels as named. Fractions, ratios and
shares are dimensionless.

## Contour and object schemas

| File | Columns and units |
| --- | --- |
| `curvature/contour_fit_points.csv` | `point_index_contour_order` (ordered integer identity); `x,y` (pixels); `theta_unwrapped` (radians); `r_pixel,r_fit_pixel,radial_residual_pixel` (pixels); `radial_residual_mm` (mm); `curvature_pixel` (pixel⁻¹); `curvature_1_per_mm` (mm⁻¹). The residual-mm field belongs to the source schema; the loader explicitly checks the other listed fields. |
| `curvature/ideal_curve.csv` | `theta` (radians); `r_pixel,x,y` (pixels); `curvature_pixel` (pixel⁻¹). The study table contains the recorded 20,000 ideal-spiral samples used for path exposure. |
| `curvature/contour_object_mapping.csv` | `feature_class` (`Detach`, `Internal` or `Surface`); `object_index`; `legacy_nearest_contour_index`; `legacy_mapped_x,legacy_mapped_y,legacy_distance_px,legacy_r_pixel` (pixels); `legacy_distance_mm,legacy_r_mm` (mm); `legacy_curvature_1_per_mm` (mm⁻¹); `legacy_curvature_bin` (categorical). `legacy_` is the historical prefix for the study's fixed-center assignment. |
| `objects/*_object_mapping.csv` | `feature_class,object_index`; `Volume3d,volume_voxel` (voxel counts); `volume_mm3` (mm³); `BaryCenterX,BaryCenterY,BaryCenterZ` (pixel/slice indices). Each class table must match its contour assignments in row count and object-index order. |

`curvature/fit_source_metadata.json` records `a_pixels,b_pixels` (the
Archimedean parameters in pixel units, with radians dimensionless),
`center_pixels` (two image coordinates), `legacy_parameter_covariance_pixel`
(2×2 covariance in pixel²), and descriptive `status,input_level` strings.
The rebuild entry compares its fit with `a_pixels,b_pixels`. The fixed-center
driver carries the covariance into uncertainty diagnostics.

`objects/source_geometry.json` supplies a `transform` object describing
upstream image preparation: `original_shape,crop_bbox_xyxy,resize_size_wh,
pad_tblr,dilate_kernel,output_size_wh,affine_original_xy_to_output_xy,dtype`.
Shape, bounding-box, padding and kernel dimensions use pixels; the affine
array is a homogeneous 3×3 image-coordinate transform. The driver copies the
object into output metadata without applying it to new images.

## Cached angular measurements

These external CSVs contain upstream measurements from segmented slices.
The driver recalculates the fitted-path denominator while retaining the voxel
counts. The original height window and image frame must agree with the tables.

| File | Source columns |
| --- | --- |
| `regions/angle_region_comparison.csv` | `scope,start_file,end_file,n_slices,axial_span_mm,region,pixel_count_per_slice,geometric_exposure_pixel_count,geometric_exposure_voxel_count,geometric_exposure_volume_mm3,positive_voxel_count,positive_volume_mm3,class_scope_volume_share,annulus_volume_share,geometric_pixel_share_of_annulus,occupancy_fraction,volume_share_to_geometric_pixel_share,path_proxy_is_unit_path_and_not_cathode_mask,path_proxy_thickness_mm,path_proxy_axial_span_mm` |
| `regions/angle_scope_summary.csv` | `scope,start_file,end_file,n_slices,axial_span_mm,foreground_voxel_count_scope,foreground_intensity_sum_scope,annulus_voxel_count,annulus_volume_mm3,sector_voxel_count,sector_volume_mm3,complement_voxel_count,complement_volume_mm3,sector_share_of_annulus_volume,complement_share_of_annulus_volume,peak_angle_bin_start_deg,peak_angle_bin_end_deg,peak_angle_bin_voxel_count,peak_angle_bin_share_of_annulus` |
| `regions/angle_5deg_histogram.csv` | `scope,start_file,end_file,n_slices,angle_bin_index,angle_bin_start_deg,angle_bin_end_deg,angle_bin_center_deg,positive_voxel_count,share_of_annulus` |
| `regions/angle_slice_counts.csv` | `scope,scope_start_file,scope_end_file,file_number,filename,foreground_voxel_count,foreground_intensity_sum,annulus_voxel_count,annulus_intensity_sum,sector_voxel_count,sector_intensity_sum,complement_voxel_count,complement_intensity_sum` |

The adapter requires `scope,region,n_slices,annulus_volume_share,
positive_volume_mm3,path_proxy_thickness_mm,path_proxy_axial_span_mm` in the
region-comparison table, where `region` is `sector` or `complement`. It
preserves other source fields and copies the other three CSVs into outputs.

File indices are integer slice identifiers with inclusive window limits.
`*_pixel_count*` and `*_voxel_count*` are counts; `*_volume_mm3` are mm³;
`*_span_mm,*_thickness_mm` are mm; `*_deg` are degrees. Intensity sums
retain the original image scale. Shares/occupancies are dimensionless and
the path-proxy flag is boolean. A unit-path proxy is a normalization
construction rather than a segmented material mask.

## Axial profiles and validation counts

`axial/axial_profiles.csv` has exactly 1,590 common rows. Columns are
`axial_index,detach_slice_index` (ordered integer indices),
`detach_position_mm` (mm), `node_nonzero_row_fraction` and
`Detach_fullheight_defect_fraction` (dimensionless fractions). Preserve the
acquisition order. Node analysis uses the fractions; the equal-height summary
also reports `axial_index`.

`validation/per_case_counts.json` has `metric_per_case` (a list) and `mean`
(an object). Each case contains `metrics` keyed by `"1"` (Detach), `"2"`
(Internal) and `"3"` (Surface). Each label contains integer `TP,FP,FN,TN`
counts and dimensionless `Dice,IoU`. Undefined case metrics are JSON `null`.
`mean[label]` supplies stored `Dice,IoU` for consistency checks. Descriptive
`case_index,aggregation,classes` fields may be present. The script reaggregates
saved counts and loads no framework, checkpoint or prediction image.

## Mechanics files and column order

```text
mechanics/
  geometry_config.json
  published_parameters.json
  curvature_summary.json
  bending/
    D_offset_buffer_k450_surfaces.txt
    D_offset_buffer_k450_balance.txt
    D_reference_buffer_k450_surfaces.txt
    D_reference_buffer_k450_balance.txt
  end_bending/
    <case>_matrix.txt
    <case>_interface.txt
    <case>_surfaces.txt
    <case>_increments.txt
  cumulative_transfer/
    stage_1turn_finite_end_restraint_ell_5mm_profile.txt
    stage_3turn_finite_end_restraint_ell_5mm_profile.txt
    stage_6turn_finite_end_restraint_ell_5mm_profile.txt
    full_finite_end_restraint_ell_5mm_profile.txt
```

The four case prefixes are `F_end_bending_base,F_zero_increment_check,
Figure6F_end_bending_mismatch,F_end_bending_normal_free`. External
`<case>_reactions.txt` exports may exist but are not read by this driver.

TXT files are whitespace-delimited numeric exports. `%` lines are comments.
Readers use column position, not header names. Preserve this order:

| Table | Numeric columns in order |
| --- | --- |
| Bending surfaces | `X_m,Y_m,u_m,v_m,sx_Pa,sy_Pa,sxy_Pa,stress_scale_Pa` |
| Bending balance | `kappa_per_m,axial_relax,total_right_reaction_N` |
| End-bending matrix | `X_m,Y_m,sx_Pa,sy_Pa,sxy_Pa,principal_normalized,u_m,v_m` |
| End-bending interface | `ds_m,branch,X_m,Y_m,sx_Pa,sy_Pa,sxy_Pa,nx,ny,tn_c_Pa,ts_c_Pa,tn_a_Pa,ts_a_Pa,scale_Pa,u_m,v_m` |
| End-bending surfaces | `X_m,Y_m,u_m,v_m,sx_Pa,sy_Pa,sxy_Pa,exx,eyy,gxy` |
| End-bending increments | `X_m,Y_m,u_m,v_m,ub_m,vb_m` |
| Cumulative profiles | `X,s_mm,theta_rad,r_mm,Psi,Psi_comsol,U,U_minus_Psi,U_X,G_mm,v_mm,slip_mm,N_over_Aeq,tau_over_Aeq_per_mm,beta` |

Local coordinates/displacements are meters, stresses/scales pascals,
reactions newtons and curvature m⁻¹. `axial_relax` is dimensionless;
`ds_m` is an arc-length weight in meters; `branch` identifies the interface
side; `nx,ny` are unit-normal components. `tn_c,ts_c,tn_a,ts_a` are
cathode/aluminum normal/shear tractions. `principal_normalized` is the
dimensionless positive tensile principal stress. Strains `exx,eyy,gxy` are
dimensionless. All matrix cases must share identical sample coordinates.
`ub_m,vb_m` are stored base displacements for compatibility subtraction.

For cumulative profiles, `X=s/S,Psi,Psi_comsol,U,U_minus_Psi,U_X,beta`
are dimensionless; `theta_rad` is radians; `s_mm,r_mm,G_mm,v_mm,slip_mm`
are mm. `N_over_Aeq` is dimensionless; `tau_over_Aeq_per_mm` is mm⁻¹.
The reader assigns the canonical force name by position if an old header
incorrectly adds `_per_mm`. All 15 columns must be finite.

### Mechanics JSON fields

`geometry_config.json` supplies `vertices_mm` (ordered polygon vertices),
`domain_mm` (length and height), `node_x_intervals_mm`,
`ordinary_x_intervals_mm` (lists of inclusive intervals), and
`matrix_neighborhood_distance_mm`. All these lengths are mm. Other geometry
construction fields may be present. The polygon/intervals must match exports.

`published_parameters.json` supplies `local_domain_mm` (length and section
thickness, mm), `cathode_E_MPa` (MPa),
`nominal_bending_curvature_per_mm` (mm⁻¹), and `F_gamma0` (dimensionless
strain). Other original material, geometry and winding parameters may be
present but are not required by the strain/normalization entry.

`curvature_summary.json` contains a `fit` object carried into cumulative
output as geometry provenance. The original object includes `a_pixel,b_pixel`
(pixels), `a_mm,b_mm` (mm), angle bounds/spans (radians), point counts,
parameter standard deviations and residual diagnostics in their named units.
Calculations use supplied profile values without refitting this metadata.

`axial_edge_theory.py` needs no input file. It requires `--height-mm`,
`--transfer-length-mm` and dimensionless `--strain-mismatch`; `--points`
sets the sample count and `--output` the CSV path. The caller explicitly
supplies these axial parameters.

## Image inputs and configuration

`configs/image_analysis.json` contains original method settings. Run, for
example:

```console
python code/analysis/run_image_tools.py flatten --data-root /external/data --output-root /external/results
```

`--config` selects a JSON with the same task sections. Its relative `inputs`
paths are resolved beneath the data root, even if the configuration is
elsewhere. Absolute paths are accepted. Defaults are:

| Input key | Path beneath data root | Method |
| --- | --- | --- |
| `collector_source_tif` | `analysis/images/collector_source.tif` | Collector |
| `collector_png` | `analysis/images/collector_reference.png` | Collector reference/vertical |
| `surface_in_png` | `analysis/images/surface_crack_flattened_inside.png` | Flatten |
| `detach_out_png` | `analysis/images/detachment_flattened_outside.png` | Density |
| `detach_dir` | `external/segmented/Detach` | Vertical |
| `internal_crack_dir` | `external/segmented/InternalCrack` | Radial/angle |

Supply one 2D binary/grayscale TIFF per axial slice for stack methods in
the configured orientation/frame. Filenames are naturally sorted. Flattened
images and collector maps are prepared upstream. Thresholds define foreground
according to each method. Set `collector.reference_input` to `null` to
disable its optional reference. Null exposure inputs retain the configured
geometric exposure definition.

General keys are `voxel_size_mm` (mm), `seed` (integer),
`bootstrap_samples` (count), and `block_length` (profile samples). Task
sections specify spacing (mm), image-intensity thresholds, resize/kernel
dimensions and lag limits (pixels), physical lag/radius limits (mm where
named), angular bins (degrees), profile bins/Gaussian widths (mm), and
permutations. Suffixes `_px,_mm,_deg` identify units. Original seeds and
processing settings are preserved in the source configuration.

`--input` overrides a primary image/stack; `--collector` overrides the
vertical reference. Relative overrides use the working directory.
`--output-root` selects the parent of `analysis/images/<task>/`;
`--output-dir` takes precedence for the full task directory. Images remain
external and are read only.
