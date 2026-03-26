"""
Convert COCO format dataset to YOLO format for RF-DETR training.

This script converts the vineyard COCO dataset to YOLO format which RF-DETR expects.
"""

import json
import os
import shutil
from pathlib import Path


def convert_coco_to_yolo_annotations(coco_json_path, output_txt_dir, coco_img_dir):
    """
    Convert COCO JSON annotations to YOLO format (.txt files).
    
    Args:
        coco_json_path: Path to _annotations.coco.json file
        output_txt_dir: Directory to save .txt annotation files
        coco_img_dir: Directory containing images
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create annotation directory if it doesn't exist
    os.makedirs(output_txt_dir, exist_ok=True)
    
    # Build image info lookup
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Build category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Convert to YOLO format
    converted_count = 0
    for img_id, anns in image_annotations.items():
        img_info = image_info[img_id]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Create txt file with same name as image
        txt_filename = Path(img_filename).stem + '.txt'
        txt_path = os.path.join(output_txt_dir, txt_filename)
        
        with open(txt_path, 'w') as txt_f:
            for ann in anns:
                cat_id = ann['category_id'] - 1  # YOLO uses 0-indexed categories
                bbox = ann['bbox']  # [x, y, width, height] in COCO format
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height]
                # Normalize to 0-1 range
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                # Clamp values to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                txt_f.write(f"{cat_id} {x_center} {y_center} {width} {height}\n")
        
        converted_count += 1
    
    return converted_count


def create_data_yaml(dataset_base, num_classes, class_names):
    """
    Create data.yaml for RF-DETR training.
    
    Args:
        dataset_base: Base directory path
        num_classes: Number of object classes
        class_names: List of class names
    """
    yaml_content = f"""
path: {dataset_base}
train: images/train
val: images/val
test: images/test

nc: {num_classes}
names: {class_names}
"""
    
    yaml_path = os.path.join(dataset_base, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    return yaml_path


def convert_vineyard_dataset(dataset_base):
    """
    Convert entire vineyard dataset from COCO to YOLO format.
    
    Args:
        dataset_base: Base directory of vineyard dataset
    """
    class_names = ["vineyard", "pole", "trunk", "vine_row"]
    num_classes = len(class_names)
    
    print("=" * 80)
    print("Converting Vineyard Dataset from COCO to YOLO Format")
    print("=" * 80)
    
    splits = ['train', 'valid', 'test']
    total_converted = 0
    
    for split in splits:
        split_dir = os.path.join(dataset_base, split)
        coco_json = os.path.join(split_dir, '_annotations.coco.json')
        
        if not os.path.exists(coco_json):
            print(f"⚠️  {split} annotations not found: {coco_json}")
            continue
        
        # Create labels directory
        labels_dir = os.path.join(split_dir, 'labels')
        
        # Also create images directory structure if needed
        images_dir = split_dir
        
        print(f"\nConverting {split} split...")
        print(f"  Source: {coco_json}")
        print(f"  Output: {labels_dir}")
        
        converted = convert_coco_to_yolo_annotations(coco_json, labels_dir, split_dir)
        print(f"  ✓ Converted {converted} annotations")
        
        total_converted += converted
    
    # Create data.yaml
    print(f"\nCreating data.yaml...")
    yaml_path = create_data_yaml(dataset_base, num_classes, class_names)
    print(f"✓ Created: {yaml_path}")
    
    print("\n" + "=" * 80)
    print(f"Dataset conversion complete!")
    print(f"Total annotations converted: {total_converted}")
    print(f"Dataset is now ready for RF-DETR training")
    print("=" * 80)
    
    return yaml_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert COCO format dataset to YOLO format for RF-DETR"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../../data/datasets/datasets_coco/vineyard_segmentation_paper-2",
        help="Path to vineyard dataset base directory"
    )
    
    args = parser.parse_args()
    
    dataset_base = os.path.abspath(args.dataset)
    
    if not os.path.exists(dataset_base):
        print(f"Error: Dataset directory not found: {dataset_base}")
        exit(1)
    
    convert_vineyard_dataset(dataset_base)
