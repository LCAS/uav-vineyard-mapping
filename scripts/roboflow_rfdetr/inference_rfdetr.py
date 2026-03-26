"""
RF-DETR Inference Script for Vineyard Detection

This script performs inference using a trained RF-DETR model to detect
poles, trunks, and vine rows in vineyard images.

Usage:
    python inference_rfdetr.py --model <path_to_model> --input <image_or_folder> --output <output_dir>
"""

import os
import sys
import torch
import cv2
import json
from pathlib import Path
import argparse
from typing import List, Dict, Tuple

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import RF-DETR models
try:
    from rfdetr import (
        RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase,
        RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge, RFDETRSegXLarge
    )
except ImportError:
    print("Error: rfdetr package not installed.")
    print("Install with: pip install rfdetr")
    sys.exit(1)


def get_model_class(model_size: str, segmentation: bool = False):
    """
    Get the appropriate RF-DETR model class based on size.
    
    Args:
        model_size: 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge/base)
        segmentation: If True, return segmentation model variants
    
    Returns:
        Model class
    """
    model_size = model_size.lower()
    
    if segmentation:
        models = {
            's': RFDETRSegSmall,
            'm': RFDETRSegMedium,
            'l': RFDETRSegLarge,
            'x': RFDETRSegXLarge,
        }
    else:
        models = {
            's': RFDETRSmall,
            'm': RFDETRMedium,
            'l': RFDETRLarge,
            'x': RFDETRBase,  # XLarge variant uses RFDETRBase
        }
    
    if model_size not in models:
        raise ValueError(f"Invalid model_size: {model_size}. Options: s, m, l, x")
    
    return models[model_size]


class VineyardDetector:
    """Wrapper class for RF-DETR inference on vineyard images."""
    
    # Class names mapping
    CLASS_NAMES = {
        0: "vineyard",
        1: "pole",
        2: "trunk",
        3: "vine_row"
    }
    
    # Colors for visualization (BGR format)
    CLASS_COLORS = {
        0: (255, 0, 0),      # vineyard - blue
        1: (0, 255, 0),      # pole - green
        2: (0, 0, 255),      # trunk - red
        3: (255, 255, 0)     # vine_row - cyan
    }
    
    def __init__(self, model_path: str = None, model_size: str = "l", device: str = "auto", conf_threshold: float = 0.5):
        """
        Initialize the vineyard detector.
        
        Args:
            model_path: Path to the trained RF-DETR model checkpoint (optional)
            model_size: Model size if loading from pretrained ('s', 'm', 'l', 'x')
            device: Device to use ('cuda', 'cpu', or 'auto')
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading RF-DETR model (size: {model_size})...")
        model_class = get_model_class(model_size, segmentation=False)
        self.model = model_class()
        
        # Load checkpoint if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading checkpoint from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.model.load_state_dict(checkpoint['model'])
            elif isinstance(checkpoint, dict):
                self.model.model.load_state_dict(checkpoint)
            else:
                self.model.model.load_state_dict(checkpoint)
        
        # RF-DETR handles device placement internally
        print("Model loaded successfully!")
    
    def predict(self, image_path: str) -> List[Dict]:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detections with format:
            [{'class_id': int, 'class_name': str, 'confidence': float, 
              'bbox': [x1, y1, x2, y2]}, ...]
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Run inference - RF-DETR returns (boxes, scores, labels)
        results = self.model.predict(image, conf_threshold=self.conf_threshold)
        
        # Format detections
        detections = []
        
        # Handle different RF-DETR output formats
        if isinstance(results, tuple):
            # Format: (boxes, scores, labels)
            boxes, scores, labels = results
            for box, score, label in zip(boxes, scores, labels):
                detection = {
                    'class_id': int(label),
                    'class_name': self.CLASS_NAMES.get(int(label), f"class_{int(label)}"),
                    'confidence': float(score),
                    'bbox': [float(x) for x in box]  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        elif isinstance(results, list):
            # Format: list of dicts or other format
            for det in results:
                if isinstance(det, dict):
                    detection = {
                        'class_id': det.get('class_id', det.get('label', 0)),
                        'class_name': self.CLASS_NAMES.get(det.get('class_id', det.get('label', 0)), 'unknown'),
                        'confidence': det.get('confidence', det.get('score', 0.0)),
                        'bbox': det.get('bbox', det.get('box', [0, 0, 0, 0]))
                    }
                    detections.append(detection)
        
        return detections
    
    def visualize(self, image_path: str, detections: List[Dict], 
                  save_path: str = None, show: bool = False) -> None:
        """
        Visualize detections on image.
        
        Args:
            image_path: Path to input image
            detections: List of detections from predict()
            save_path: Path to save annotated image (optional)
            show: Whether to display image (optional)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Draw detections
        for det in detections:
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Get color for this class
            color = self.CLASS_COLORS.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1, label_size[1] + 10)
            
            cv2.rectangle(image, (x1, label_y - label_size[1] - 10),
                         (x1 + label_size[0], label_y), color, -1)
            cv2.putText(image, label, (x1, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save image if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)
            print(f"Saved annotated image to: {save_path}")
        
        # Show image if requested
        if show:
            cv2.imshow("Vineyard Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def save_detections_json(self, detections: List[Dict], output_path: str) -> None:
        """
        Save detections to JSON file.
        
        Args:
            detections: List of detections from predict()
            output_path: Path to save JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f"Saved detections to: {output_path}")


def process_single_image(detector: VineyardDetector, image_path: str, 
                        output_dir: str, save_json: bool = True,
                        visualize: bool = True) -> Dict:
    """
    Process a single image.
    
    Args:
        detector: VineyardDetector instance
        image_path: Path to input image
        output_dir: Directory to save outputs
        save_json: Whether to save detections as JSON
        visualize: Whether to save visualized image
        
    Returns:
        Dictionary with detection statistics
    """
    print(f"\nProcessing: {image_path}")
    
    # Run inference
    detections = detector.predict(image_path)
    
    # Count detections by class
    class_counts = {}
    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"  Total detections: {len(detections)}")
    for class_name, count in class_counts.items():
        print(f"    {class_name}: {count}")
    
    # Save outputs
    image_name = Path(image_path).stem
    
    if save_json:
        json_path = os.path.join(output_dir, "json", f"{image_name}.json")
        detector.save_detections_json(detections, json_path)
    
    if visualize:
        viz_path = os.path.join(output_dir, "visualizations", f"{image_name}_annotated.jpg")
        detector.visualize(image_path, detections, save_path=viz_path)
    
    return {
        'image': image_path,
        'total_detections': len(detections),
        'class_counts': class_counts,
        'detections': detections
    }


def process_folder(detector: VineyardDetector, input_folder: str, 
                   output_dir: str, save_json: bool = True,
                   visualize: bool = True) -> List[Dict]:
    """
    Process all images in a folder.
    
    Args:
        detector: VineyardDetector instance
        input_folder: Folder containing input images
        output_dir: Directory to save outputs
        save_json: Whether to save detections as JSON
        visualize: Whether to save visualized images
        
    Returns:
        List of results for each image
    """
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return []
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]")
        result = process_single_image(detector, image_path, output_dir, 
                                     save_json, visualize)
        results.append(result)
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")
    
    # Print overall statistics
    total_detections = sum(r['total_detections'] for r in results)
    print("\n" + "=" * 80)
    print("Overall Statistics")
    print("=" * 80)
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    
    # Aggregate class counts
    all_class_counts = {}
    for result in results:
        for class_name, count in result['class_counts'].items():
            all_class_counts[class_name] = all_class_counts.get(class_name, 0) + count
    
    for class_name, count in all_class_counts.items():
        print(f"  {class_name}: {count}")
    print("=" * 80)
    
    return results


def main():
    """
    Main inference function.
    
    Example usage:
    
    # Single image inference
    python inference_rfdetr.py \\
        --model runs/detect/vineyard_segmentation_paper_2_l/weights/best.pth \\
        --input /path/to/image.jpg \\
        --output inference_results/single_image \\
        --conf 0.5
    
    # Folder inference with dynamic output path (Riseholme example)
    IMAGE_DATE="july_2025"
    ALTITUDE=39
    MODEL_NAME="vineyard_segmentation_paper_2_l"
    IMAGE_SIZE="1280x960"
    
    python inference_rfdetr.py \\
        --model runs/detect/${MODEL_NAME}/weights/best.pth \\
        --input ../../../images/riseholme/${IMAGE_DATE}/${ALTITUDE}_feet/ \\
        --output inference_results/rfdetr_test/${MODEL_NAME}/${IMAGE_DATE}/${ALTITUDE}_feet/image_size_${IMAGE_SIZE} \\
        --conf 0.5 \\
        --model-size l
    
    # This creates organized output structure:
    # inference_results/rfdetr_test/vineyard_segmentation_paper_2_l/july_2025/39_feet/image_size_1280x960/
    #   ├── json/                    # Detection results as JSON
    #   ├── visualizations/          # Annotated images
    #   └── summary.json             # Overall statistics
    """
    parser = argparse.ArgumentParser(
        description="RF-DETR inference for vineyard detection"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained RF-DETR model"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input image or folder"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="../../data/output/rfdetr_inference",
        help="Output directory for results"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="l",
        choices=["s", "m", "l", "x"],
        help="Model size (s/m/l/x, default: l)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto, default: auto)"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON output"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display images (only for single image mode)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    detector = VineyardDetector(
        model_path=args.model,
        model_size=args.model_size,
        device=args.device,
        conf_threshold=args.conf
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        print("\n" + "=" * 80)
        print("Processing single image")
        print("=" * 80)
        process_single_image(
            detector, args.input, args.output,
            save_json=not args.no_json,
            visualize=not args.no_viz
        )
        
        if args.show:
            detections = detector.predict(args.input)
            detector.visualize(args.input, detections, show=True)
    
    elif os.path.isdir(args.input):
        # Folder of images
        print("\n" + "=" * 80)
        print("Processing folder of images")
        print("=" * 80)
        process_folder(
            detector, args.input, args.output,
            save_json=not args.no_json,
            visualize=not args.no_viz
        )
    
    else:
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    print("\nInference completed!")


if __name__ == "__main__":
    # ======================
    # CONFIGURATION MODE
    # ======================
    # Uncomment and edit this section to run without command-line arguments.
    # Comment out "main()" below and uncomment the configuration section.
    
    # --- Configuration ---
    USE_CONFIG = True  # Set to True to use configuration below instead of CLI args
    
    if USE_CONFIG:
        # Edit these variables to match your setup
        IMAGE_DATE = "august_2024"  # Options: "july_2025", "august_2024", "march_2025"
        ALTITUDE = 65  # Options: 39, 65, 100 (feet)
        MODEL_NAME = "vineyard_segmentation_paper_2_l"
        IMAGE_SIZE = "1280x960"  # For output folder naming
        
        # Paths
        MODEL_PATH = f"runs/detect/{MODEL_NAME}/weights/best.pth"
        INPUT_DIR = f"../../images/riseholme/{IMAGE_DATE}/{ALTITUDE}_feet/"
        OUTPUT_DIR = f"inference_results/rfdetr_test/{MODEL_NAME}/{IMAGE_DATE}/{ALTITUDE}_feet/image_size_{IMAGE_SIZE}"
        
        # Inference parameters
        MODEL_SIZE = "l"  # s, m, l, or x
        CONF_THRESHOLD = 0.5
        DEVICE = "auto"  # "cuda", "cpu", or "auto"
        SAVE_VIZ = True
        SAVE_JSON = True
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Initialize detector
        print(f"Running inference with configuration:")
        print(f"  Model: {MODEL_PATH}")
        print(f"  Input: {INPUT_DIR}")
        print(f"  Output: {OUTPUT_DIR}")
        print(f"  Confidence: {CONF_THRESHOLD}")
        
        detector = VineyardDetector(
            model_path=MODEL_PATH,
            model_size=MODEL_SIZE,
            device=DEVICE,
            conf_threshold=CONF_THRESHOLD
        )
        
        # Process input
        if os.path.isfile(INPUT_DIR):
            # Single image
            print("\n" + "=" * 80)
            print("Processing single image")
            print("=" * 80)
            process_single_image(
                detector, INPUT_DIR, OUTPUT_DIR,
                save_json=SAVE_JSON,
                visualize=SAVE_VIZ
            )
        elif os.path.isdir(INPUT_DIR):
            # Folder of images
            print("\n" + "=" * 80)
            print("Processing folder of images")
            print("=" * 80)
            process_folder(
                detector, INPUT_DIR, OUTPUT_DIR,
                save_json=SAVE_JSON,
                visualize=SAVE_VIZ
            )
        else:
            print(f"Error: Input path does not exist: {INPUT_DIR}")
            sys.exit(1)
        
        print("\nInference completed!")
    
    else:
        # Use command-line arguments (default)
        main()
