# Scripts Guide

This directory is the canonical home for all Python entrypoints in the public repository.

## Active Core Workflows

These paths make up the supported public workflow surface:

- `cluster_poles.py`: standalone duplicate-detection clustering utility
- `run_clustering_example.py`: runnable example using the bundled Riseholme sample output
- `generate_topological_map.py`: topological-map export entrypoint
- `generate_topo_map/`: interactive inference and map-generation tooling
- `topological_map_scripts/`: topological-map helpers used by the export pipeline
- `gaussian_heatmap_resnet/`: main ResNet and hybrid paper workflow code
- root-level helper scripts such as `image_gps_pixel_show_poles.py`, `poles_to_rows.py`, `mid_row_lines.py`, and `pole_clustering.py`

## Optional Toolkits

These areas are retained because they are documented and potentially useful, but they are not part of the minimal public workflow:

- `gcp_calibration/`: camera calibration and annotation toolkit
- `roboflow_rfdetr/`: RF-DETR training and inference toolkit

## Archive

Archived code lives under `archive/`. It is kept for provenance, comparison, and internal experimentation, but it is not part of the supported public workflow.

- `archive/apple_depth_pro/`
- `archive/graph_matching/`
- `archive/pygmtools_graph_matching/`
- `archive/roboflow_scripts/`
- `archive/gaussian_heatmap_resnet_legacy/`

See [archive/README.md](/home/pulver/projects/vineyard_detection/uav-vineyard-mapping/scripts/archive/README.md) for the archive policy.
