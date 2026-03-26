"""
Flask app for annotating ground control points in images.

Allows you to:
1. View images from a folder
2. Click on image to mark point locations
3. Select which control point it is
4. Save results as GCP JSON files for calibration
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from pathlib import Path
import base64
from io import BytesIO

app = Flask(__name__)

# Configuration - use absolute path to avoid issues with working directory
IMAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../images/riseholme/august_2024"))
IMAGE_FOLDERS = ["39_feet", "65_feet", "100_feet"]
IMAGE_FOLDER_PATHS = {name: os.path.join(IMAGE_ROOT, name) for name in IMAGE_FOLDERS}
OUTPUT_FOLDER = "gcp_annotations"

# Control points data
CONTROL_POINTS = {
    "1": {"name": "4", "lat": 53.26821989, "lon": -0.524252789, "elevation": 84.78154407},
    "2": {"name": "1", "lat": 53.26802554, "lon": -0.524166502, "elevation": 83.86054286},
    "3": {"name": "3", "lat": 53.26799377, "lon": -0.524593067, "elevation": 83.97875},
    "4": {"name": "2", "lat": 53.2681784, "lon": -0.524652258, "elevation": 85.51847},
}

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get list of images
def get_image_list():
    """Get list of image files from all folders"""
    image_extensions = {".jpg", ".jpeg", ".png"}
    images = []

    for folder_name, folder_path in IMAGE_FOLDER_PATHS.items():
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                images.append(f"{folder_name}/{filename}")

    return sorted(images)


def resolve_image_path(image_key):
    """Resolve a folder/filename key to an absolute image path safely."""
    parts = image_key.split("/", 1)
    if len(parts) != 2:
        return None

    folder_name, filename = parts
    folder_path = IMAGE_FOLDER_PATHS.get(folder_name)
    if not folder_path:
        return None

    safe_name = os.path.basename(filename)
    if safe_name != filename:
        return None

    return os.path.join(folder_path, safe_name)


def safe_output_stem(image_key):
    """Create a filesystem-safe stem for output JSON files."""
    return image_key.replace("/", "__")

# Store annotations in memory (can be persisted to JSON)
annotations = {}

@app.route('/')
def index():
    """Main page"""
    images = get_image_list()
    return render_template('gcp_annotator.html', images=images, control_points=CONTROL_POINTS)

@app.route('/api/images')
def api_images():
    """Get list of images"""
    images = get_image_list()
    return jsonify(images)

@app.route('/api/debug')
def api_debug():
    """Debug endpoint to check configuration"""
    images = get_image_list()
    return jsonify({
        "image_root": IMAGE_ROOT,
        "image_folders": IMAGE_FOLDERS,
        "image_folder_exists": {name: os.path.exists(path) for name, path in IMAGE_FOLDER_PATHS.items()},
        "output_folder": os.path.abspath(OUTPUT_FOLDER),
        "total_images": len(images),
        "first_images": images[:5] if images else [],
        "flask_debug": app.debug
    })

@app.route('/api/image/<path:filename>')
def api_get_image(filename):
    """Get image data as base64"""
    filepath = resolve_image_path(filename)

    if not filepath or not os.path.exists(filepath):
        print(f"❌ Image not found: {filename}")
        return jsonify({"error": "Image not found", "path": filename}), 404
    
    try:
        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        print(f"✓ Loaded image: {filename} ({len(image_data)} bytes)")
        
        return jsonify({
            "filename": filename,
            "data": f"data:image/jpeg;base64,{image_data}",
            "annotations": annotations.get(filename, [])
        })
    except Exception as e:
        print(f"❌ Error loading image {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/annotations/<path:filename>', methods=['POST'])
def api_save_annotations(filename):
    """Save annotations for an image"""
    data = request.json
    annotations[filename] = data.get('points', [])
    return jsonify({"status": "saved"})

@app.route('/api/annotations/<path:filename>', methods=['GET'])
def api_get_annotations(filename):
    """Get annotations for an image"""
    return jsonify(annotations.get(filename, []))

@app.route('/api/export/<path:filename>')
def api_export_gcps(filename):
    """Export annotations as GCP JSON file"""
    if filename not in annotations or not annotations[filename]:
        return jsonify({"error": "No annotations for this image"}), 404
    
    # Get image path to extract EXIF
    image_path = resolve_image_path(filename)
    if not image_path:
        return jsonify({"error": "Invalid image path"}), 400
    
    # Build GCP JSON
    gcps_data = {
        "image_path": os.path.relpath(image_path),
        "image_filename": filename,
        "gcps": []
    }
    
    for point in annotations[filename]:
        if point.get('control_point'):
            cp_id = point['control_point']
            cp_data = CONTROL_POINTS[cp_id]
            
            gcps_data["gcps"].append({
                "pixel_x": int(point['x']),
                "pixel_y": int(point['y']),
                "gps_lat": cp_data['lat'],
                "gps_lon": cp_data['lon'],
                "label": f"Control Point {cp_data['name']}",
                "elevation": cp_data['elevation']
            })
    
    # Save to file
    output_path = os.path.join(OUTPUT_FOLDER, f"{safe_output_stem(filename)}_gcps.json")
    with open(output_path, 'w') as f:
        json.dump(gcps_data, f, indent=2)
    
    return jsonify({
        "status": "exported",
        "path": output_path,
        "gcps_count": len(gcps_data["gcps"])
    })

@app.route('/api/export-all')
def api_export_all():
    """Export all annotations as GCP JSON files"""
    exported_files = []
    
    for filename, points in annotations.items():
        if not points:
            continue

        image_path = resolve_image_path(filename)
        if not image_path:
            continue
        
        gcps_data = {
            "image_path": os.path.relpath(image_path),
            "image_filename": filename,
            "gcps": []
        }
        
        for point in points:
            if point.get('control_point'):
                cp_id = point['control_point']
                cp_data = CONTROL_POINTS[cp_id]
                
                gcps_data["gcps"].append({
                    "pixel_x": int(point['x']),
                    "pixel_y": int(point['y']),
                    "gps_lat": cp_data['lat'],
                    "gps_lon": cp_data['lon'],
                    "label": f"Control Point {cp_data['name']}",
                    "elevation": cp_data['elevation']
                })
        
        if gcps_data["gcps"]:
            output_path = os.path.join(OUTPUT_FOLDER, f"{safe_output_stem(filename)}_gcps.json")
            with open(output_path, 'w') as f:
                json.dump(gcps_data, f, indent=2)
            exported_files.append(output_path)
    
    return jsonify({
        "status": "exported",
        "files": exported_files,
        "count": len(exported_files)
    })

@app.route('/api/stats')
def api_stats():
    """Get annotation statistics"""
    total_images = len(get_image_list())
    annotated_images = len([f for f in annotations if annotations[f]])
    total_points = sum(len(points) for points in annotations.values())
    
    return jsonify({
        "total_images": total_images,
        "annotated_images": annotated_images,
        "total_points": total_points
    })

@app.route('/api/delete-annotation/<path:filename>', methods=['DELETE'])
def api_delete_annotation(filename):
    """Delete annotation for an image"""
    if filename in annotations:
        del annotations[filename]
    return jsonify({"status": "deleted"})

if __name__ == '__main__':
    print(f"📁 Image root: {os.path.abspath(IMAGE_ROOT)}")
    print(f"📂 Image folders: {', '.join(IMAGE_FOLDERS)}")
    print(f"📊 Found {len(get_image_list())} images")
    print(f"💾 Output folder: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"\n🌐 Starting Flask server...")
    print(f"   Local:   http://localhost:5005")
    print(f"   Network: http://<your-machine-ip>:5005")
    print(f"\n   To find your machine IP, run: hostname -I")
    print(f"   Press Ctrl+C to stop\n")
    # Bind to 0.0.0.0 to allow network access from other machines
    app.run(host='0.0.0.0', debug=True, port=5005)
