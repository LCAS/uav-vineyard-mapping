# Curated Results

This directory contains the curated experiment outputs kept in the public release.

## Included

- YOLO object-detection summary CSVs
- YOLO segmentation summary CSVs
- ResNet summary CSVs
- localisation reports used during map-accuracy analysis
- workbook snapshots retained for reference (`results/workbooks/`)
- provenance/config files copied from representative training and inference runs

## Excluded

- `.docx` notes and planning files
- `old/`, `wrong/`, and duplicate result exports
- bulk per-image inference dumps and large generated GeoJSON forests

## Directory Guide

- `yolo_object_detection/`: main object-detection summaries, cross-altitude, cross-seasonality, and localisation CSVs
- `yolo_segmentation/`: main segmentation summaries, cross-altitude, cross-seasonality, and localisation CSVs
- `resnets/`: ResNet family summaries plus localisation CSVs
- `workbooks/`: the original `Results.xlsx` files retained as supplementary reference material
- `provenance/`: representative config and metrics files retained to document how key results were produced

## Provenance Notes

The provenance files are copied from the original experiment workspace without rewriting their internal paths. They are included as a record of the original training/inference settings rather than as polished end-user configuration files.
