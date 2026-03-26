import os
import cv2
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import json

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_PATH = "../../data/datasets/trained/vineyard_segmentation-22/train5/weights/best.pt"

IMAGE_DIR = "../../images/agri_tech_centre/jojo"
OUTPUT_DIR = "../../data/datasets/auto_labeled_yolo_posts/bbox/agri_tech_centre/jojo/"

# IMAGE_DIR = "../../images/jojo/riccardo/DJI_202507311147_029_jojo3-120"
# OUTPUT_DIR = "../../data/datasets/auto_labeled_yolo_posts/bbox/riccardo/jojo/DJI_202507311147_029_jojo3-120/"

# YOLO class IDs (update based on your model)
CLASS_NAMES = {
    0: "pole",  # Only interested in posts
    # Add other classes if needed
}

# Label format: 'bbox' or 'segmentation'
LABEL_FORMAT = 'bbox'  # Change to 'segmentation' for polygon labels

# SAHI parameters
SLICE_HEIGHT = 1520  # 640
SLICE_WIDTH = 2028  # 640
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2
CONFIDENCE_THRESHOLD = 0.4

# Output options
COPY_IMAGES = False  # Set to False to only create label files without copying images

# ==========================================================
# UTILITIES
# ==========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def create_yolo_bbox_label(detection, img_width, img_height):
    """
    Convert detection to YOLO bbox format: class x_center y_center width height (normalized)
    """
    bbox = detection.bbox
    x_min, y_min, x_max, y_max = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
    
    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize by image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    class_id = detection.category.id
    
    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"


def mask_to_polygon_coordinates(mask):
    """
    Convert binary mask to polygon coordinates for YOLO segmentation format.
    Returns normalized coordinates as a flat list.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour to reduce points
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Flatten and return normalized coordinates
    points = approx.reshape(-1, 2)
    return points


def create_yolo_segmentation_label(detection, img_width, img_height):
    """
    Convert detection to YOLO segmentation format: class x1 y1 x2 y2 ... xn yn (normalized)
    """
    if not hasattr(detection, 'mask') or detection.mask is None:
        # Fall back to bbox if no mask
        return create_yolo_bbox_label(detection, img_width, img_height)
    
    mask = detection.mask.bool_mask
    polygon_points = mask_to_polygon_coordinates(mask)
    
    if polygon_points is None:
        return create_yolo_bbox_label(detection, img_width, img_height)
    
    # Normalize coordinates
    normalized_coords = []
    for x, y in polygon_points:
        normalized_coords.append(f"{x/img_width:.6f}")
        normalized_coords.append(f"{y/img_height:.6f}")
    
    class_id = detection.category.id
    coords_str = " ".join(normalized_coords)
    
    return f"{class_id} {coords_str}"


def process_image(image_path, model, output_images_dir, output_labels_dir):
    """
    Run SAHI inference on image and generate YOLO labels.
    """
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # Run SAHI sliced prediction
    result = get_sliced_prediction(
        str(image_path),
        model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
    )
    
    # Filter detections: only posts with confidence > threshold
    post_detections = [
        det for det in result.object_prediction_list
        if det.category.name == "pole" and det.score.value > CONFIDENCE_THRESHOLD
    ]
    
    if len(post_detections) == 0:
        print(f"[INFO] No posts detected in {os.path.basename(image_path)}")
        return
    
    # Generate label file
    image_name = os.path.basename(image_path)
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(output_labels_dir, label_name)
    
    with open(label_path, 'w') as f:
        for detection in post_detections:
            if LABEL_FORMAT == 'segmentation':
                label_line = create_yolo_segmentation_label(detection, img_width, img_height)
            else:  # bbox
                label_line = create_yolo_bbox_label(detection, img_width, img_height)
            
            f.write(label_line + "\n")
    
    # Copy image to output directory if enabled
    if COPY_IMAGES:
        import shutil
        shutil.copy(image_path, os.path.join(output_images_dir, image_name))
    
    print(f"[OK] {image_name}: {len(post_detections)} posts")


# ==========================================================
# MAIN
# ==========================================================
def main():
    # Setup output directories
    images_dir = os.path.join(OUTPUT_DIR, "images")
    labels_dir = os.path.join(OUTPUT_DIR, "labels")
    ensure_dir(images_dir)
    ensure_dir(labels_dir)
    
    # Load YOLO model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device="cuda:0"  # Change to "cpu" if no GPU
    )
    
    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
        image_files.extend(Path(IMAGE_DIR).glob(ext))
    
    print(f"Found {len(image_files)} images")
    print(f"Label format: {LABEL_FORMAT}")
    print(f"Processing...")
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {img_path.name}...")
        try:
            process_image(str(img_path), detection_model, images_dir, labels_dir)
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path.name}: {e}")
    
    # Create data.yaml for Roboflow/YOLOv8
    data_yaml_content = f"""# Dataset configuration for YOLOv8
path: {os.path.abspath(OUTPUT_DIR)}
train: images
val: images  # You should split this into train/val/test

# Classes
names:
  0: post

# Number of classes
nc: 1
"""
    
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n✅ Dataset created in: {OUTPUT_DIR}")
    print(f"   - Images: {len(list(Path(images_dir).glob('*')))} files")
    print(f"   - Labels: {len(list(Path(labels_dir).glob('*.txt')))} files")
    print(f"   - Format: {LABEL_FORMAT}")
    print(f"\n📁 To import to Roboflow:")
    print(f"   1. Zip the '{OUTPUT_DIR}' folder")
    print(f"   2. Upload to Roboflow (YOLOv8 format)")
    print(f"   3. Split into train/val/test sets in Roboflow")


if __name__ == "__main__":
    main()
