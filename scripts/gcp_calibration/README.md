# 🎯 GCP Calibration Toolkit

Complete toolkit for calibrating drone camera parameters using Ground Control Points.

## 📁 Folder Structure

```
gcp_calibration/
├── calibrate_camera_with_gcps.py          ⭐ Main calibration optimizer
├── gcp_image_annotator.py                 🎨 Web UI for marking control points
├── ground_control_points_example.json      📋 Example GCP file format
├── gcp_annotations/                        📁 Output folder (auto-created)
│   ├── image1_gcps.json
│   ├── image2_gcps.json
│   └── ...
├── templates/
│   └── gcp_annotator.html                 🌐 Web interface
├── run_gcp_annotator.sh                   🚀 Quick start script
├── gcp_annotator_requirements.txt          📦 Python dependencies
├── GCP_ANNOTATOR_README.md                📖 Web UI guide
├── GCP_CALIBRATION_README.md              📖 Calibration guide
└── README.md                              📄 This file
```

## 🚀 Quick Start

### Option 1: Annotate Control Points (Web UI)

```bash
# From this folder
bash run_gcp_annotator.sh
```

Then open **http://localhost:5000** in your browser to:
1. View images
2. Click on control points
3. Select which control point (1-4) it is
4. Export as GCP JSON files

📖 Full guide: [GCP_ANNOTATOR_README.md](GCP_ANNOTATOR_README.md)

### Option 2: Run Calibration

```bash
# After exporting GCPs from the web UI
python calibrate_camera_with_gcps.py

# Or manually edit the script to load your JSON files
```

The calibrator outputs optimized camera parameters:
```
FOCAL_LENGTH_MM = 4.523
SENSOR_WIDTH_MM = 6.245
SENSOR_HEIGHT_MM = 4.612
```

📖 Full guide: [GCP_CALIBRATION_README.md](GCP_CALIBRATION_README.md)

## 📋 Workflow

```
1. Run Web UI
   └─> bash run_gcp_annotator.sh

2. Annotate Images
   └─> Click points, select control points, export GCPs

3. Calibrate Camera
   └─> python calibrate_camera_with_gcps.py

4. Use Results
   └─> Copy parameters to inference_segmentation_yolo_labels_full.py
```

## 🎯 Your Control Points

```
Point 1: Lat 53.26821989, Lon -0.524252789
Point 2: Lat 53.26802554, Lon -0.524166502
Point 3: Lat 53.26799377, Lon -0.524593067
Point 4: Lat 53.2681784,  Lon -0.524652258
```

## 📦 Installation

```bash
pip install -r gcp_annotator_requirements.txt
```

## 🔗 Integration

After calibration, update your inference script:

```python
# inference_segmentation_yolo_labels_full.py
FOCAL_LENGTH_MM = 4.523      # ← From calibration
SENSOR_WIDTH_MM = 6.245
SENSOR_HEIGHT_MM = 4.612
```

## 📚 Documentation

- **[GCP_ANNOTATOR_README.md](GCP_ANNOTATOR_README.md)** - Web UI guide
- **[GCP_CALIBRATION_README.md](GCP_CALIBRATION_README.md)** - Calibration guide
- **[ground_control_points_example.json](ground_control_points_example.json)** - Example format

## 🔧 Files Overview

### Core Files
| File | Purpose |
|------|---------|
| `calibrate_camera_with_gcps.py` | Optimize camera parameters using GCPs |
| `gcp_image_annotator.py` | Flask web app for annotating points |
| `templates/gcp_annotator.html` | Interactive web interface |

### Configuration
| File | Purpose |
|------|---------|
| `ground_control_points_example.json` | Example GCP JSON format |
| `gcp_annotator_requirements.txt` | Python dependencies |
| `run_gcp_annotator.sh` | Quick start script |

### Documentation
| File | Purpose |
|------|---------|
| `GCP_ANNOTATOR_README.md` | Web UI instructions |
| `GCP_CALIBRATION_README.md` | Calibration process guide |
| `README.md` | This file |

## 🎨 Web UI Features

✅ Image browsing  
✅ Click-to-mark control points  
✅ Visual feedback with color-coded points  
✅ Control point assignment  
✅ Keyboard navigation (arrow keys)  
✅ Live statistics  
✅ Bulk export to JSON  

## 🧪 Output Format

Each annotated image generates a calibration-ready JSON:

```json
{
  "image_path": "../../images/...",
  "image_filename": "DJI_*.JPG",
  "gcps": [
    {
      "pixel_x": 2028,
      "pixel_y": 1520,
      "gps_lat": 53.268188,
      "gps_lon": -0.524277,
      "label": "Control Point 1"
    }
  ]
}
```

## 💡 Tips

1. **Annotate multiple images** - More data = better calibration
2. **Spread points across image** - Corners, edges, and center
3. **Use global search** for best accuracy if local search doesn't work well
4. **Check RMSE** - Aim for < 0.5m for best results

## 🆘 Troubleshooting

### "No images found"
- Check image folder path: `../../../images/riseholme/august_2024/39_feet/`
- Ensure images are JPG/PNG format

### High RMSE during calibration
- Verify GPS coordinates are correct
- Add more GCPs (minimum 3, recommended 5+)
- Try global optimization

### Flask won't start
- Install dependencies: `pip install -r gcp_annotator_requirements.txt`
- Check port 5000 is not in use

## 📞 Support

Refer to individual README files for detailed documentation:
- Web UI issues → [GCP_ANNOTATOR_README.md](GCP_ANNOTATOR_README.md)
- Calibration issues → [GCP_CALIBRATION_README.md](GCP_CALIBRATION_README.md)

---

**Ready to calibrate?** Start with: `bash run_gcp_annotator.sh`
