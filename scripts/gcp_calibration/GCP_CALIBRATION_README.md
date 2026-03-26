# Camera Calibration with Ground Control Points (GCPs)

## Overview
This tool helps you find the accurate camera parameters (focal length, sensor width/height) by using ground control points - known GPS locations visible in your images.

## Problem
The current scripts use estimated values for:
- `FOCAL_LENGTH_MM` 
- `SENSOR_WIDTH_MM`
- `SENSOR_HEIGHT_MM`

These estimates can cause GPS positioning errors. With ground control points, we can optimize these values.

## Quick Start

### 1. Identify Ground Control Points
Find markers/objects in your image where you know:
- ✅ The **pixel coordinates** (x, y) in the image
- ✅ The **GPS coordinates** (latitude, longitude)

**Tip:** Use GIMP, Photoshop, or any image viewer to find pixel coordinates. You need at least 2-3 GCPs, but more is better (5-10 recommended).

### 2. Prepare Your GCP Data

**Important:** All GCPs must come from the same image. The image path is set in Step 3.

#### Option A: Edit the Python script directly
```python
calibrator.add_gcp(pixel_x, pixel_y, gps_lat, gps_lon, "Label")
calibrator.add_gcp(2028, 1520, 53.26815000, -0.52457500, "Pole 1")
calibrator.add_gcp(1500, 1000, 53.26818842, -0.52427737, "Pole 2")
calibrator.add_gcp(3456, 1890, 53.26808856, -0.52425335, "Pole 3")
```

#### Option B: Use a JSON file (recommended)
Create `my_gcps.json` with the image path included:
```json
{
  "image_path": "../../images/riseholme/august_2024/39_feet/DJI_20240802143112_0076_W.JPG",
  "gcps": [
    {
      "pixel_x": 2028,
      "pixel_y": 1520,
      "gps_lat": 53.26815000,
      "gps_lon": -0.52457500,
      "label": "Pole 1"
    },
    {
      "pixel_x": 1500,
      "pixel_y": 1000,
      "gps_lat": 53.26818842,
      "gps_lon": -0.52427737,
      "label": "Pole 2"
    },
    {
      "pixel_x": 3456,
      "pixel_y": 1890,
      "gps_lat": 53.26808856,
      "gps_lon": -0.52425335,
      "label": "Pole 3"
    }
  ]
}
```

Then load it in your script:
```python
calibrator = GCPCalibrator.from_json("my_gcps.json")
```

### 3. Configure the Script

#### Option A: Load from JSON file (simplest for multiple images)
```python
import calibrate_camera_with_gcps as gcp

# Load everything from JSON (image path + all GCPs)
calibrator = gcp.GCPCalibrator.from_json("my_gcps.json")
```

#### Option B: Manual configuration
Edit `calibrate_camera_with_gcps.py`:

```python
# Set the image path and initial guesses
IMAGE_PATH = "../../images/riseholme/august_2024/39_feet/DJI_20240802143112_0076_W.JPG"
INITIAL_FOCAL_LENGTH = 4.5
INITIAL_SENSOR_WIDTH = 6.17
INITIAL_SENSOR_HEIGHT = 4.55

calibrator = GCPCalibrator(IMAGE_PATH)

# Add GCPs from this image
calibrator.add_gcp(2028, 1520, 53.26815000, -0.52457500, "Pole 1")
calibrator.add_gcp(1500, 1000, 53.26818842, -0.52427737, "Pole 2")
```

### 4. Run Calibration

```bash
cd /home/cheddar/code/vineyard_detection/scripts/gaussian_heatmap_resnet
python calibrate_camera_with_gcps.py
```

### 5. Use the Results

The script will output optimized values:
```
🎯 CALIBRATED CAMERA PARAMETERS
==========================================
FOCAL_LENGTH_MM = 4.523
SENSOR_WIDTH_MM = 6.245
SENSOR_HEIGHT_MM = 4.612
==========================================
```

**Copy these values** into your [inference_segmentation_yolo_labels_full.py](inference_segmentation_yolo_labels_full.py):

```python
# Camera Specs (Riseholme H20) - CALIBRATED
FOCAL_LENGTH_MM = 4.523
SENSOR_WIDTH_MM = 6.245
SENSOR_HEIGHT_MM = 4.612
```

## Optimization Methods

### Method 1: Local Search (Fast) ⚡
- Good when initial estimates are close
- Fast (~seconds)
- May find local minimum

```python
results = calibrator.calibrate_local_search(
    initial_params=(4.5, 6.17, 4.55),
    param_ranges=((3.0, 6.0), (4.0, 8.0), (3.0, 6.0))
)
```

### Method 2: Global Search (Thorough) 🌍
- Searches entire parameter space
- More robust, finds global minimum
- Slower (~1-2 minutes)

```python
results = calibrator.calibrate_global_search(
    param_ranges=((3.0, 6.0), (4.0, 8.0), (3.0, 6.0))
)
```

## Tips for Best Results

1. **More GCPs = Better accuracy**
   - Minimum: 2-3 GCPs
   - Recommended: 5-10 GCPs
   - Optimal: 10-20 GCPs

2. **Spread GCPs across the image**
   - Don't cluster all points in one area
   - Cover corners and edges
   - Include center points

3. **Use high-confidence points**
   - Clear, identifiable markers
   - Accurate GPS measurements
   - Avoid blurry or ambiguous locations

4. **Verify GPS coordinates**
   - Double-check lat/lon values
   - Ensure proper decimal precision (6-8 digits)
   - Watch for sign errors (N/S, E/W)

## Understanding the Output

```
📊 Validation Results:
   GCP 1: Pole A
     Actual:    (53.26818842, -0.52427737)
     Predicted: (53.26818845, -0.52427740)
     Error:     0.035m

   RMSE: 0.087m
```

- **RMSE (Root Mean Square Error)**: Average positioning error
  - < 0.5m: Excellent
  - 0.5-1.0m: Good
  - 1.0-2.0m: Acceptable for many applications
  - > 2.0m: Check your GCPs and try global optimization

## Troubleshooting

### High RMSE (> 2m)
- ✓ Verify GPS coordinates are correct
- ✓ Check pixel coordinates match the GPS locations
- ✓ Ensure image EXIF data is intact
- ✓ Try global optimization
- ✓ Add more GCPs

### "No GPS data found"
- ✓ Use an image with EXIF metadata intact
- ✓ Some editors strip EXIF data - use original drone images

### Parameters seem unrealistic
- ✓ Widen the search ranges in `param_ranges`
- ✓ Check camera specifications online
- ✓ Try global optimization

## Example Workflows

### Workflow 1: Using JSON file (recommended for multiple images)
```python
import calibrate_camera_with_gcps as gcp

# 1. Load calibrator from JSON (includes image path + GCPs)
calibrator = gcp.GCPCalibrator.from_json("image1_gcps.json")

# 2. Run optimization
results = calibrator.calibrate_local_search(
    initial_params=(4.5, 6.17, 4.55)
)

# 3. Validate and save
calibrator.validate_parameters(
    results['focal_length_mm'],
    results['sensor_width_mm'],
    results['sensor_height_mm']
)
calibrator.save_results(results, "camera_params_image1.json")
```

### Workflow 2: Manual configuration
```python
# 1. Create calibrator
calibrator = GCPCalibrator("my_image.jpg")

# 2. Add GCPs
calibrator.add_gcp(1234, 567, 53.268188, -0.524277, "Corner Post")
calibrator.add_gcp(2345, 1234, 53.268156, -0.524301, "Center Marker")
calibrator.add_gcp(3456, 1890, 53.268123, -0.524325, "Far Post")

# 3. Run optimization
results = calibrator.calibrate_local_search(
    initial_params=(4.5, 6.17, 4.55)
)

# 4. Validate and save
calibrator.validate_parameters(
    results['focal_length_mm'],
    results['sensor_width_mm'],
    results['sensor_height_mm']
)
calibrator.save_results(results, "camera_params.json")
```

## Files Created

- `calibrate_camera_with_gcps.py` - Main calibration script
- `ground_control_points_example.json` - Example GCP file (with image_path included)
- `camera_calibration_results.json` - Output file with results

**For multiple images, create separate JSON files:**
- `image1_gcps.json` - GCPs from image 1 (includes image_path)
- `image2_gcps.json` - GCPs from image 2 (includes image_path)
- etc.

## Next Steps

After calibration:
1. ✅ Copy the optimized parameters to your inference script
2. ✅ Run inference and check improved accuracy
3. ✅ Optionally calibrate for different altitudes or camera angles
4. ✅ Save calibration results for different camera/drone setups

## References

- Original scripts: `image_gps_pixel_show_poles.py`, `inference_segmentation_yolo_labels_full.py`
- Optimization: Uses `scipy.optimize` (L-BFGS-B and Differential Evolution)
