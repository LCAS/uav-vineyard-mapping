"""
RF-DETR Training Script for Vineyard Detection

This script trains an RF-DETR model to detect poles, trunks, and vine rows
in drone images of vineyards using the COCO-formatted dataset.

Dataset classes:
- vineyard (id=0)
- pole (id=1)
- trunk (id=2)
- vine_row (id=3)
"""

import os
import sys
import torch
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

# Apply supervision compatibility patches before importing RF-DETR
try:
    import supervision_patch  # noqa: F401 - Apply monkey patches
except ImportError:
    pass  # If patch isn't available, RF-DETR might still work

# Import RF-DETR models (install with: pip install rfdetr)
try:
    from rfdetr import (
        RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase,
        RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge, RFDETRSegXLarge
    )
    from rfdetr.config import TrainConfig
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


def main():
    """
    Main training function for RF-DETR on vineyard detection dataset.
    """
    
    # Dataset configuration - use absolute path
    # The vineyard dataset should be in COCO format initially
    # Run `python coco_to_yolo.py --create-yaml` first to convert it to YOLO format
    dataset_base = os.path.abspath("../../data/datasets/datasets_coco/vineyard_segmentation_paper-2")
    
    if not os.path.exists(dataset_base):
        print(f"Error: Dataset not found at {dataset_base}")
        print("\nPlease ensure the vineyard dataset is at:")
        print("  /home/cheddar/code/vineyard_detection/data/datasets/datasets_coco/vineyard_segmentation_paper-2")
        sys.exit(1)
    
    # Check if dataset has been converted to YOLO format
    train_labels_dir = os.path.join(dataset_base, "train", "labels")
    if not os.path.exists(train_labels_dir):
        print("Error: YOLO format labels not found.")
        print(f"  Expected at: {train_labels_dir}")
        print("\nPlease run the converter first:")
        print("  python coco_to_yolo.py --dataset <dataset_path> --create-yaml")
        sys.exit(1)
    test_dir = os.path.join(dataset_base, "test")
    
    # Training configuration using RF-DETR TrainConfig
    model_size = "l"  # Options: 's', 'm', 'l', 'x'
    
    # Create YOLO-style output directory structure
    # Format: runs/detect/{dataset}_{modelsize}/
    dataset_name = os.path.basename(os.path.abspath(dataset_base)).replace('-', '_')
    run_name = f"{dataset_name}_{model_size}"
    output_base = "runs/detect"
    output_dir = os.path.join(output_base, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    weights_dir = os.path.join(output_dir, "weights")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("RF-DETR Training Configuration")
    print("=" * 80)
    print(f"Dataset: {dataset_base}")
    print(f"Model size: {model_size}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Early Stopping: ENABLED (patience=10 epochs)")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Initialize model
    print("\nInitializing RF-DETR model...")
    model_class = get_model_class(model_size, segmentation=False)
    model = model_class()
    print(f"✓ Model initialized: {model_class.__name__}")
    
    # Create training configuration
    print("\nCreating training configuration...")
    train_config = TrainConfig(
        dataset_dir=dataset_base,
        dataset_file="yolo",  # YOLO dataset format with train/valid/test dirs
        output_dir=output_dir,
        
        # Training hyperparameters
        epochs=100,
        batch_size=4,  # Adjust based on GPU memory
        grad_accum_steps=2,  # Gradient accumulation for memory efficiency
        lr=1e-4,  # Learning rate
        lr_encoder=1.5e-4,  # Slightly higher for encoder
        weight_decay=1e-4,
        
        # Warmup and scheduling
        warmup_epochs=5,
        lr_drop=100,
        
        # Early stopping - IMPLEMENTED
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        
        # Checkpointing
        checkpoint_interval=10,
        
        # Data loading
        num_workers=4,
        
        # Class names (vineyard dataset has 4 classes)
        class_names=["vineyard", "pole", "trunk", "vine_row"],
        
        # EMA (Exponential Moving Average)
        use_ema=True,
        ema_decay=0.993,
        
        # Logging
        tensorboard=True,
        project="vineyard_detection",
        run=run_name,
    )
    
    # Save configuration to YAML file
    config_path = os.path.join(output_dir, "args.yaml")
    with open(config_path, 'w') as f:
        yaml.dump({
            'epochs': train_config.epochs,
            'batch_size': train_config.batch_size,
            'lr': train_config.lr,
            'weight_decay': train_config.weight_decay,
            'warmup_epochs': train_config.warmup_epochs,
            'early_stopping': train_config.early_stopping,
            'early_stopping_patience': train_config.early_stopping_patience,
            'dataset': dataset_base,
            'model_size': model_size,
            'output_dir': output_dir,
        }, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Configuration saved to {config_path}")
    
    # Start training
    print("\nStarting training...")
    print(f"Training on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 80)
    print()
    
    try:
        model.train_from_config(train_config)
    except Exception as e:
        print(f"\n⚠️  Training error: {e}")
        print("\nNote: RF-DETR training requires dataset in a specific format.")
        print("The dataset_dir should contain train/valid/test subdirectories.")
        print("Each should have images and corresponding YOLO-format annotations.")
        print("\nAlternatively, ensure th dataset is in standard COCO/Roboflow format.")
        print("For more details, refer to: https://github.com/roboflow/rf-detr")
        return output_dir
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Save training metadata
    metadata = {
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_base,
        "dataset_name": dataset_name,
        "run_name": run_name,
        "output_directory": output_dir,
        "model_size": model_size,
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Training Summary for: {run_name}")
    print("=" * 80)
    print(f"Output Directory: {output_dir}/")
    print(f"  ├── checkpoints/      (model checkpoints)")
    print(f"  ├── args.yaml         (all training parameters)")
    print(f"  ├── metadata.json     (training metadata)")
    print(f"  └── runs/             (tensorboard logs)")
    print("=" * 80)
    print("\nTo use the trained model for inference:")
    print(f"  python inference_rfdetr.py --model_size {model_size} --input <image_or_folder>")
    print("=" * 80)
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RF-DETR model on vineyard dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--model-size", type=str, default="l", choices=['s', 'm', 'l', 'x'],
                        help="Model size: s=small, m=medium, l=large, x=xlarge")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    main()
