# Data Manifest

This repository does not ship the raw UAV datasets used for training and evaluation. Public datasets are referenced externally to keep the GitHub repository lightweight and to avoid duplicating archival releases.

## Public Dataset Releases

### Riseholme COCO-format multi-season dataset

- DOI: `10.5281/zenodo.19234907`
- Citation title: "Multi-Temporal UAV Vineyard Segmentation Dataset: Riseholme (COCO Format, 2024--2025)"

Expected local extraction layout:

- `data/dataset/riseholme-august-2024-full-resolution.coco-segmentation/`
- `data/dataset/riseholme-march-2025-full-resolution.coco-segmentation/`
- `data/dataset/riseholme-july-2025-full-resolution.coco-segmentation/`

### AGRIDS UAV vineyard dataset release

- DOI: `10.5281/zenodo.15211733`
- Citation title: "AGRIDS: UAV vineyard image detection dataset"

Use this release when working with the broader AGRIDS/YOLO-oriented dataset material referenced by the manuscript.

## Included In GitHub

- lightweight examples under `examples/`
- curated result summaries under `results/`
- public Riseholme ground-truth references under `ground_truth/riseholme/`

## Intentionally Excluded From GitHub

- raw UAV imagery
- Roboflow downloads and export caches
- large COCO/YOLO dataset folders
- partner-site and non-public field data
- local API keys and secrets
- trained weights and large intermediate outputs

## Local Path Contract

Several scripts in `scripts/` are research-oriented and expect local paths such as:

- `images/...`
- `weights/...`
- `config/api_key.json`

Those locations are user-supplied and are not part of this public repository.
