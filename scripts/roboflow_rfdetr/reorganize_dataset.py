"""
Reorganize YOLO dataset structure for RF-DETR compatibility.

RF-DETR expects:
  dataset/
  ├── train/
  │   ├── images/  (all image files)
  │   └── labels/  (all .txt label files)
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def reorganize_split(split_dir, split_name):
    """
    Reorganize a dataset split into RF-DETR compatible structure.
    
    Args:
        split_dir: Path to train/valid/test directory
        split_name: 'train', 'valid', or 'test'
    """
    images_dir = os.path.join(split_dir, 'images')
    labels_dir_new = os.path.join(split_dir, 'labels_temp')
    labels_dir_old = os.path.join(split_dir, 'labels')
    
    # Create images directory
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir_new, exist_ok=True)
    
    print(f"\n{split_name.upper()} set:")
    print(f"  Moving images to {images_dir}/")
    
    # Move image files to images/
    image_count = 0
    for filename in tqdm(os.listdir(split_dir)):
        filepath = os.path.join(split_dir, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
        
        # Check if it's an image file
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            dest = os.path.join(images_dir, filename)
            shutil.move(filepath, dest)
            image_count += 1
    
    print(f"  ✓ Moved {image_count} images")
    
    # Move label files to new labels directory
    if os.path.exists(labels_dir_old):
        print(f"  Moving labels to {labels_dir_new}/")
        label_count = 0
        for filename in tqdm(os.listdir(labels_dir_old)):
            if filename.endswith('.txt'):
                src = os.path.join(labels_dir_old, filename)
                dest = os.path.join(labels_dir_new, filename)
                shutil.copy(src, dest)
                label_count += 1
        
        print(f"  ✓ Copied {label_count} labels")
        
        # Remove old labels directory
        shutil.rmtree(labels_dir_old)
    
    # Rename labels_temp to labels
    if os.path.exists(labels_dir_new):
        if os.path.exists(os.path.join(split_dir, 'labels')):
            shutil.rmtree(os.path.join(split_dir, 'labels'))
        os.rename(labels_dir_new, os.path.join(split_dir, 'labels'))


def reorganize_dataset(dataset_base):
    """
    Reorganize entire dataset for RF-DETR compatibility.
    
    Args:
        dataset_base: Base directory containing train/, valid/, test/
    """
    print("=" * 80)
    print("Reorganizing Dataset for RF-DETR Compatibility")
    print("=" * 80)
    print(f"Dataset: {dataset_base}")
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_dir = os.path.join(dataset_base, split)
        if os.path.exists(split_dir):
            reorganize_split(split_dir, split)
        else:
            print(f"⚠️  {split.upper()} directory not found: {split_dir}")
    
    print("\n" + "=" * 80)
    print("✓ Dataset reorganization complete!")
    print("Dataset structure is now compatible with RF-DETR")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reorganize dataset for RF-DETR")
    parser.add_argument("--dataset", type=str,
                        default="../../data/datasets/datasets_coco/vineyard_segmentation_paper-2",
                        help="Path to dataset base directory")
    
    args = parser.parse_args()
    
    dataset_base = os.path.abspath(args.dataset)
    
    if not os.path.exists(dataset_base):
        print(f"Error: Dataset not found at {dataset_base}")
        exit(1)
    
    reorganize_dataset(dataset_base)
