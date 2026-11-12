# stomata-analysis
# Stomatal detection and spatial pattern analysis

Detects stomata in leaf-imprint micrographs with Mask R-CNN, measures stomatal traits, and analyses the spatial
pattern of the stomata (Ripley's K/L, Moran's I, Getis-Ord Gi\*, kernel density estimation).

Two models are used: a **CPA** model segmenting the whole stomatal complex and an **APA** model segmenting the
stomatal aperture. Guard cell area is derived as GCA = CPA − APA.

## Installation

Detection needs Python 3.8 and TensorFlow 2.5:

```
conda create -n stomata python=3.8
conda activate stomata
pip install -r requirements.txt
```

The spatial analysis (`spatial.py`) needs only numpy, scipy, pandas and matplotlib and also runs in recent
Python versions.

## Model weights

Download the trained weights from the **Releases** page of this repository and put them in `dnn_model/`:

| File | Model |
|---|---|
| `stomatacomplexarea.h5` | CPA (stomatal complex) |
| `stomatalAPA.h5` | APA (stomatal aperture) |

The models were trained on *Lilium* leaf imprints (10×/0.65 objective, 2880 × 2048 px images). For other species
or imaging set-ups, check the detections (`save_overlays`) and retrain if needed (see *Training your own models*).

## Quick start

1. Arrange the images by group and sampling unit:

   ```
   images/
     control/          <- group (e.g. genotype, treatment)
       plant01/        <- sampling unit (e.g. plant); its images are replicates
         img1.jpg
         img2.jpg
       plant02/
     treated/
       plant01/
   ```

   Images placed directly in a group folder are treated as one unit of that group.

2. Copy `config_example.json` to `config.json` and set at least `images_folder` and `pixels_per_um`.

3. Run:

   ```
   python detect.py config.json     # detection and trait measurement
   python spatial.py config.json    # spatial analysis
   ```

Relative paths in the configuration file are taken relative to the location of the configuration file.

## Configuration file

| Key | Meaning |
|---|---|
| `images_folder` | Folder with the images (`<group>/<unit>/<image>`) |
| `output_folder` | Folder for the results |
| `pixels_per_um` | **Pixel scale** of the images, pixels per µm (calibrate with a stage micrometer) |
| `weights` | Weight files of the `CPA` and `APA` models; leave one empty (`null`) to run only the other |
| `group_weights` | Optional other weights for some groups, e.g. `{"speciesB": {"CPA": "...", "APA": "..."}}` |
| `network_input_size` | Size to which images are resized inside the network (832, as in training); any image size can be used |
| `detection_min_confidence` | Minimum detection score (0.7) |
| `save_overlays` | Save the images with the detections drawn (CPA green, APA red) |
| `spatial.seed` | Random seed of the simulations and permutations |
| `spatial.min_stomata_per_image` | Images with fewer stomata are skipped in the spatial analyses (5) |
| `spatial.compare_groups` | Pairs of groups to compare, e.g. `[["control", "treated"]]`; empty = all pairs |
| `ripley.max_distance_um`, `step_um` | Distances at which L(r) is evaluated |
| `ripley.n_simulations` | CSR simulations for the global envelope test |
| `ripley.n_permutations` | Permutations of the studentized permutation test between groups |
| `ripley.size_null_model` | Also test against a size-aware (hard-core) null model (assumes the stomata are aligned with the image x axis) |
| `moran.neighbourhoods`, `gistar.neighbourhoods` | Neighbourhood definitions: `knnK` (K nearest neighbours), `bandM` (within M × mean nearest-neighbour distance), `delaunay` |
| `moran.n_permutations` | Permutations for the Moran's I P value |
| `gistar.z_threshold` | \|z\| above which a point is a hot or cold spot (1.65 = 90% level) |
| `gistar.n_null_permutations` | Within-image permutations giving the expected fraction of flagged points without spatial structure |
| `kde.bandwidth` | Bandwidth factor of `scipy.stats.gaussian_kde` (or `"scott"`) |
| `kde.grid_size`, `kde.threshold` | Grid of the KDE maps; normalised density above which a cell counts as high density |

Each analysis can be switched off with `"run": false`.

## Output

`detect.py` writes to `output_folder`:

| File | Content |
|---|---|
| `stomata.csv` | One row per detection: group, unit, image, model (CPA/APA), centroid (px, µm), area (µm²), length and width (µm; sides of the minimum-area rectangle), score, bounding box |
| `images.csv` | One row per image: size (px, µm), number of detections, stomatal density (CPA detections per mm²) |
| `paired.csv` | One row per stoma detected by both models (paired by the nearest mask centroid): CPA, APA, GCA (µm²), complex and aperture length and width, junction length (complex length − aperture length), APA/CPA area ratio |
| `traits_units.csv` | Unit means (mean of the image means) of density and of the traits in `paired.csv` |
| `overlays/` | Images with the detections drawn |

`spatial.py` writes to `output_folder/spatial/`:

| File | Content |
|---|---|
| `ripley_units.csv` | Per unit: number of images and stomata, density, mean nearest-neighbour distance, global envelope test P against complete spatial randomness, RPSD (smallest distance at which the pattern is more regular than random), RPSD\* = RPSD·√λ (density-free) and, if requested, the comparison with the size-aware null model |
| `ripley_curves.npz` | Mean L(r), envelopes and density-scaled L\*(r\*) of every unit |
| `ripley_tests.csv` | Group comparisons: Welch and Mann–Whitney tests of the unit values; studentized permutation tests of the L(r) and L\*(r\*) curves |
| `ripley_L.png` | Mean L(r) ± SE over units per group |
| `moran_images.csv`, `moran_units.csv`, `moran_tests.csv` | Moran's I of CPA and APA per image, unit means, group comparisons |
| `gistar_images.csv`, `gistar_units.csv`, `gistar_groups.csv`, `gistar_tests.csv` | Fraction of hot/cold spots per image and under the permutation null, excess over the null per unit and group, group comparisons |
| `kde_images.csv`, `kde_units.csv`, `kde_groups.csv`, `kde_tests.csv` | High-density extent per image and unit, high-variability extent per group, group comparisons |
| `kde_maps/` | Mean and std KDE maps per group (PNG and .npy) |

Groups are compared with the unit as the experimental unit; tests need at least two units per group.

## Using your own detections

`spatial.py` only reads `stomata.csv` and `images.csv`. Points from another detector or from manual annotation
can be analysed by writing these two files with at least the columns

- `stomata.csv`: `group, unit, image, model, cx_px, cy_px, cx_um, cy_um, area_um2, x1, y1, x2, y2`
  (`model` = `CPA` for stomatal complexes; `x1`–`y2` only for the size-aware null model)
- `images.csv`: `group, unit, image, width_px, height_px, width_um, height_um`

## Methods

The algorithms are described in the accompanying paper ([citation to be added]). Briefly:

- **Ripley's K/L**: buffer-zone edge correction (focal points at least `max_distance_um` from the image edge);
  for each unit, the images are replicate patterns and the mean L(r) is tested against the mean of CSR patterns
  simulated in the same windows with the same numbers of points, using a global rank envelope test
  (Myllymäki et al. 2017). Curves of two groups are compared with a studentized permutation test (Hahn 2012),
  on the original scale and after density scaling (r\* = r√λ, L\* = L√λ).
- **Moran's I**: row-standardised weights, permutation P value.
- **Getis-Ord Gi\***: binary weights including the focal point, analytical z-scores; the fraction of flagged
  points is compared with the fraction expected after permuting the trait values within the image.
- **KDE**: Gaussian kernel on centroids normalised to the image size, maps min–max normalised per image.

## Training your own models

One model is trained per structure (stomatal complex, aperture). Annotations are COCO-format polygon files, as
exported by most annotation tools (e.g. Makesense.ai). Copy `train_config_example.json` and set the image folder,
the annotation file(s), the output folder and a model name. A GPU is strongly recommended.

```
python train.py train_config.json
```

Semi-automated annotation, as used for the published models:

1. Annotate a few images by hand (10 were used) and train an initial model.
2. Set `autolabel_weights` to that model and run `python autolabel.py train_config.json`: all images in
   `images_folder` are detected and saved as COCO annotations (`autolabel_output`).
3. Correct these annotations in the annotation tool (remove false detections, add missed stomata, adjust
   outlines) and export them.
4. Train the final model on all corrected annotations (`annotations` can list several files).

| Key | Meaning (default = setting of the published models) |
|---|---|
| `images_folder` | Training images (looked up by the `file_name` in the annotations) |
| `annotations` | COCO JSON file or list of files |
| `output_folder`, `model_name` | Where the weights are saved after every epoch |
| `init_weights` | `"coco"` (COCO-pretrained, file in `coco_weights`), `"imagenet"`, `"last"` (continue the last run) or a weights file (e.g. one of the stomata models, to fine-tune it) |
| `validation_fraction`, `seed` | Random train / validation split (0.1) |
| `image_size`, `images_per_gpu` | Network input size (832) and batch size (2) |
| `epochs`, `steps_per_epoch`, `validation_steps` | 30, 1000, 50 |
| `learning_rate` | SGD learning rate (0.001; momentum 0.9 and weight decay 0.0001 as in Matterport) |
| `layers` | Layers trained: `"heads"`, `"all"`, `"3+"`, `"4+"`, `"5+"` |
| `augmentation` | With probability 5/6 one of: horizontal flip, vertical flip, rotation ±45°, rotation ±90°, scaling 0.5–1.5× (requires `imgaug`) |
| `autolabel_*`, `detection_min_confidence` | Weights, output file and category name for `autolabel.py` |

The COCO-pretrained weights `mask_rcnn_coco.h5` are available from the Matterport repository
(https://github.com/matterport/Mask_RCNN/releases).

## Acknowledgements

`mrcnn/` is the Matterport Mask R-CNN implementation (MIT License, https://github.com/matterport/Mask_RCNN).
