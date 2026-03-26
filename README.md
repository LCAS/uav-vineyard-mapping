# UAV Vineyard Mapping

`uav-vineyard-mapping` is a publication-facing export of the UAV vineyard mapping and robotic-navigation workspace used for the experiments reported in "An Integrated Aerial-Ground System for Vineyard Mapping and Robotic Navigation".

https://github.com/user-attachments/assets/fadc2bb0-dd91-4a22-bcf9-186bb117a595


This release focuses on:

- research code and supporting utilities
- curated CSV result summaries, workbook snapshots, and provenance files
- small public Riseholme examples
- links to public datasets hosted on Zenodo

This release intentionally does not include:

- raw training imagery
- large dataset downloads
- partner-site or non-public field data
- model weights or Roboflow API keys
- bulk intermediate inference outputs, caches, or temporary files

## Repository Layout

- `scripts/`: canonical home for all Python entrypoints, active workflows, optional toolkits, and archived research code
- `scripts/README.md`: navigation guide for active, optional, and archived script areas
- `results/`: curated experiment summaries and provenance/config files
- `examples/`: a few public sample images and lightweight output artifacts
- `ground_truth/riseholme/`: public Riseholme reference files used by several scripts
- `data/README.md`: external data contract and Zenodo download locations

## Workflow 1: Set Up The Environment

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate uav-vineyard-mapping
```

If you prefer `pip`, the `environment.yml` file also lists the main Python dependencies used across the exported workflows.

## Workflow 2: Obtain Public Data

The public datasets used in this work are hosted on Zenodo and are not committed to GitHub.

- Riseholme COCO-format multi-season dataset: `10.5281/zenodo.19234907`
- AGRIDS UAV vineyard dataset release: `10.5281/zenodo.15211733`

See [data/README.md](/home/pulver/projects/vineyard_detection/uav-vineyard-mapping/data/README.md) for the expected local folder names and what is intentionally excluded from this repository.

Important local-data assumptions:

- many training and inference scripts expect user-supplied imagery under local `images/` folders
- model checkpoints are expected under local `weights/` folders
- Roboflow-backed scripts expect a local `config/api_key.json`

Those assets are deliberately not part of this public release.

## Workflow 3: Inspect Or Reproduce Curated Results

Start with the curated outputs:

- [results/README.md](/home/pulver/projects/vineyard_detection/uav-vineyard-mapping/results/README.md)
- [examples/README.md](/home/pulver/projects/vineyard_detection/uav-vineyard-mapping/examples/README.md)

Representative entry scripts:

- `python scripts/cluster_poles.py examples/outputs/riseholme_detected_pole_coordinates.geojson`
- `python scripts/run_clustering_example.py`
- `python scripts/gaussian_heatmap_resnet/train_resnet_validation_yolo_labels.py`
- `python scripts/gaussian_heatmap_resnet/inference_segmentation_yolo_labels_full.py`
- `python scripts/generate_topological_map.py`

All Python entrypoints now live under `scripts/`. The scripts are preserved as research-oriented entrypoints rather than repackaged into a formal Python library. Some scripts use hard-coded relative paths and may need local path edits depending on how you stage imagery, weights, and API credentials.

## Notes

- The exported repository remote is configured for `https://github.com/LCAS/uav-vineyard-mapping`.
- The code is released under Apache-2.0; see [LICENSE](/home/pulver/projects/vineyard_detection/uav-vineyard-mapping/LICENSE).
