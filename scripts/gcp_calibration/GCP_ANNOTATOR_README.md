# GCP Image Annotator

A Flask web app to annotate ground control points in drone images interactively.

## Features

✅ **Image Navigation** - Browse through images in a folder  
✅ **Click Annotation** - Click on image to mark control point locations  
✅ **Point Selection** - Assign each point to a control point or skip it  
✅ **Real-time Visualization** - See marked points as you click  
✅ **Bulk Export** - Export all annotations as GCP JSON files for calibration  
✅ **Live Statistics** - Track progress across all images  

## Quick Start

### 1. Install Dependencies

```bash
pip install -r gcp_annotator_requirements.txt
```

### 2. Run the Flask App

```bash
python gcp_image_annotator.py
```

You'll see:
```
📁 Image folder: ...
📊 Found X images
💾 Output folder: ...

🌐 Starting Flask server at http://localhost:5000
```

### 3. Open in Browser

Navigate to **`http://localhost:5000`** in your web browser.

## How to Use

### Workflow

1. **View Image** - Flask loads an image from the folder
2. **Click on Points** - Click on locations in the image where you see the control points
3. **Select Control Point** - After clicking, select which control point (1, 2, 3, or 4) that location is, or click "⊘ None/Skip" if it's not a control point
4. **Review** - See the marked points in the right panel
5. **Navigate** - Use Previous/Next buttons or arrow keys to move between images
6. ****Export** - When done with an image, click "Export GCPs" or export all at the end

### Control Points

Your control points are:
- **Point 1**: Lat: 53.26821989, Lon: -0.524252789
- **Point 2**: Lat: 53.26802554, Lon: -0.524166502
- **Point 3**: Lat: 53.26799377, Lon: -0.524593067
- **Point 4**: Lat: 53.2681784, Lon: -0.524652258

### Keyboard Shortcuts

- **Left Arrow** - Previous image
- **Right Arrow** - Next image

### Output

Each annotated image produces a JSON file in the `gcp_annotations/` folder:

**Example: `DJI_20240802143112_0076_W_gcps.json`**

```json
{
  "image_path": "../../images/riseholme/august_2024/39_feet/DJI_20240802143112_0076_W.JPG",
  "image_filename": "DJI_20240802143112_0076_W.JPG",
  "gcps": [
    {
      "pixel_x": 2028,
      "pixel_y": 1520,
      "gps_lat": 53.26815000,
      "gps_lon": -0.52457500,
      "label": "Control Point 1",
      "elevation": 83.86054286
    },
    ...
  ]
}
```

## Using the Output Files

Once you've exported the GCP annotations, use them with the calibrator:

```python
from calibrate_camera_with_gcps import GCPCalibrator

# Load calibrator from the exported JSON
calibrator = GCPCalibrator.from_json("gcp_annotations/image1_gcps.json")

# Run optimization
results = calibrator.calibrate_local_search(
    initial_params=(4.5, 6.17, 4.55)
)

# Validate
calibrator.validate_parameters(
    results['focal_length_mm'],
    results['sensor_width_mm'],
    results['sensor_height_mm']
)
```

## Troubleshooting

### "No images found"
- Ensure images exist in `../../images/riseholme/august_2024/39_feet/`
- Supported formats: JPG, PNG, JPEG

### Points not showing up
- Make sure the image has finished loading (wait for image to display)
- Check that click is on the actual image area

### Export fails
- Make sure at least one control point is selected (not "None")
- Check that the output folder exists (created automatically)

## File Structure

```
gaussian_heatmap_resnet/
├── gcp_image_annotator.py           (Flask app)
├── gcp_annotator_requirements.txt    (Dependencies)
├── templates/
│   └── gcp_annotator.html           (Web UI)
├── gcp_annotations/                 (Output folder - auto-created)
│   ├── image1_gcps.json
│   ├── image2_gcps.json
│   └── ...
└── ...
```

## Keyboard Tips

- **Arrow keys** for fast navigation
- **Click multiple times** on same image if multiple points need the same control point
- **"None" button** to skip points that aren't control points
- **"Clear All"** to reset current image

## Next Steps

After annotating all images:

1. ✅ All JSON files are in `gcp_annotations/` folder
2. ✅ Use `calibrate_camera_with_gcps.py` to optimize camera parameters
3. ✅ Copy optimized parameters to your inference script

## Tips for Best Results

1. **Identify control points first** - Look for the 4 control points in each image
2. **Click accurately** - Try to click at the exact center of the control point marker
3. **One point per click** - Each click adds one annotation
4. **Use labels** - The app shows which control point you're selecting
5. **Export often** - Export GCPs for each image to avoid losing work

## Performance

- Works well with up to 100+ images
- Exports in seconds
- No server processing - all computation is local

## Browser Compatibility

Works on:
- Chrome/Chromium ✅
- Firefox ✅
- Safari ✅
- Edge ✅

Tested on desktop. Mobile/tablet may have zoom/interaction issues.

## Support

If you encounter issues:
1. Check the Flask console for error messages
2. Make sure image folder path is correct
3. Verify Flask is running (`http://localhost:5000`)
4. Try refreshing the page
