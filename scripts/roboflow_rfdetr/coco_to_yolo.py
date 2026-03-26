"""
Convert COCO format dataset to YOLO format for RF-DETR training.

The vineyard dataset is in COCO format with _annotations.coco.json files.
This script converts it to YOLO format with .txt label files.
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def coco_to_yolo(coco_json_path, images_dir, output_labels_dir):
    """
    Convert COCO annotations to YOLO format.
    
    Args:
        coco_json_path: Path to _annotations.coco.json
        images_dir: Directory containing images
        output_labels_dir: Directory to save YOLO label .txt files
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping of image_id to image info
    images = {img['id']: img for img in coco_data['images']}
    
    # Create mapping of image_id to annotations
    img_to_annot = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_annot:
            img_to_annot[img_id] = []
        img_to_annot[img_id].append(ann)
    
    # Convert each image's annotations
    converted_count = 0
    for img_id, annots in tqdm(img_to_annot.items(), desc="Converting COCO to YOLO"):
        if img_id not in images:
            continue
        
        img_info = images[img_id]
        img_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Create label file
        label_name = Path(img_name).stem + '.txt'
        label_path = os.path.join(output_labels_dir, label_name)
        
        with open(label_path, 'w') as f:
            for ann in annots:
                cat_id = ann['category_id'] - 1  # COCO uses 1-indexed, YOLO uses 0-indexed
                bbox = ann['bbox']  # [x, y, width, height]
                
                # Convert to YOLO format: [center_x, center_y, width, height] (normalized)
                x = bbox[0] / img_width
                y = bbox[1] / img_height
                w = bbox[2] / img_width
                h = bbox[3] / img_height
                
                # Center coordinates
                cx = x + w / 2
                cy = y + h / 2
                
                f.write(f"{cat_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        converted_count += 1
    
    return converted_count


def convert_dataset(dataset_base):
    """
    Convert entire COCO dataset to YOLO format.
    
    Args:
        dataset_base: Base directory containing train/, valid/, test/ subdirectories
    """
    splits = ['train', 'valid', 'test']
    
    print("=" * 80)
    print("Converting COCO Dataset to YOLO Format")
    print("=" * 80)
    
    for split in splits:
        split_dir = os.path.join(dataset_base, split)
        if not os.path.exists(split_dir):
            print(f"⚠️  {split.upper()} directory not found: {split_dir}")
            continue
        
        images_dir = split_dir
        coco_json = os.path.join(split_dir, '_annotations.coco.json')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(coco_json):
            print(f"⚠️  COCO annotations not found: {coco_json}")
            continue
        
        print(f"\n{split.upper()} set:")
        print(f"  Images: {images_dir}")
        print(f"  Annotations: {coco_json}")
        print(f"  Labels output: {labels_dir}")
        
        converted = coco_to_yolo(coco_json, images_dir, labels_dir)
        print(f"  ✓ Converted {converted} images")
    
    print("\n" + "=" * 80)
    print("✓ Dataset conversion complete!")
    print("=" * 80)


def create_data_yaml(dataset_base, output_yaml='data.yaml'):
    """
    Create data.yaml file for RF-DETR training.
    
    Args:
        dataset_base: Base directory of dataset
        output_yaml: Path to save data.yaml
    """
    data_yaml_content = f"""path: {os.path.abspath(dataset_base)}  # dataset root
train: train  # train images (relative to path)
val: valid    # val images (relative to path)
test: test    # test images (optional)

nc: 4  # number of classes
names: ['vineyard', 'pole', 'trunk', 'vine_row']  # class names
"""
    
    with open(output_yaml, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"✓ Created {output_yaml}")
    return output_yaml


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format")
    parser.add_argument("--dataset", type=str, 
                        default="../../data/datasets/datasets_coco/vineyard_segmentation_paper-2",
                        help="Path to COCO dataset base directory")
    parser.add_argument("--create-yaml", action="store_true",
                        help="Create data.yaml file")
    
    args = parser.parse_args()
    
    dataset_base = args.dataset
    
    if not os.path.exists(dataset_base):
        print(f"Error: Dataset not found at {dataset_base}")
        exit(1)
    
    # Convert dataset
    convert_dataset(dataset_base)
    
    # Create data.yaml if requested
    if args.create_yaml:
        output_yaml = os.path.join(dataset_base, 'data.yaml')
        create_data_yaml(dataset_base, output_yaml)
