# Hybrid ResNet-UNet for Vineyard Analysis

**Goal:** Simultaneously detect vineyard features with high precision (points) and robust shape extraction (areas) using a single drone image model.

### Method
* **Architecture:** A modified ResNet-18 or 50 or 101 U-Net with a dual-head output.
* **Head 1 (Regression):** Predicts **Gaussian Heatmaps** for **Poles & Trunks**.
  * *Why:* Prevents merging of close objects and allows sub-pixel center accuracy (MSE Loss).
* **Head 2 (Segmentation):** Predicts **Binary Masks** for **Vine Rows**.
  * *Why:* Captures the full polygon shape and width of the row (BCE Loss).

### Implementation Details
* **Input:** RGB Drone Imagery (640x480).
* **Outputs:**
  * **2-Channel Heatmap** (Ch0: Pole, Ch1: Trunk) -> use `argmax` for location.
  * **1-Channel Mask** (Row) -> use `findContours` for polygons.

**Usage:** Train with `train_hybrid.py`. The model learns to pinpoint poles as "glows" and paint rows as solid shapes.