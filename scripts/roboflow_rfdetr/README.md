# RF-DETR for Vineyard Detection

This directory contains training and inference scripts for the RF-DETR (Real-time DEtection TRansformer) model to detect poles, trunks, and vine rows in drone images of vineyards.

## Overview

RF-DETR is a state-of-the-art object detection model that combines the power of transformers with real-time performance. This implementation is specifically designed for vineyard detection tasks.

### Detected Classes

The model detects the following classes:
- **vineyard** (id=0) - General vineyard areas
- **pole** (id=1) - Support poles in the vineyard
- **trunk** (id=2) - Grape vine trunks
- **vine_row** (id=3) - Rows of grape vines

## Dataset

The model is trained on the dataset located at:
```
../../data/datasets/datasets_coco/vineyard_segmentation_paper-2/
```

Dataset structure:
- `train/` - Training images and annotations
- `valid/` - Validation images and annotations
- `test/` - Test images and annotations

The dataset uses COCO format annotations with 3,420 images.

## Installation

### 1. Install Dependencies

```bash
# Install RF-DETR
pip install rfdetr

# Install additional dependencies
pip install torch torchvision opencv-python pillow pyyaml
```

### 2. Verify Installation

```bash
python -c "from rfdetr import RFDETR; print('RF-DETR installed successfully!')"
```

## Usage

### Training

Train the RF-DETR model on the vineyard dataset:

```bash
# Basic training
python train_rfdetr.py

# Training with custom parameters
python train_rfdetr.py --epochs 150 --batch-size 8 --img-size 640 --model-size l --lr 1e-4
```

#### Training Parameters

- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 4)
- `--img-size`: Input image size (default: 640)
- `--model-size`: Model size - 's', 'm', 'l', 'x' (default: 'l')
  - `s` - Small: Fastest, lowest accuracy
  - `m` - Medium: Balanced
  - `l` - Large: High accuracy (recommended)
  - `x` - XLarge: Highest accuracy, slowest
- `--lr`: Learning rate (default: 1e-4)
- `--device`: Device to use - 'cuda', 'cpu', 'auto' (default: 'auto')

#### Output

Trained models are saved in:
```
../../weights/rfdetr_vineyard/
```

The directory will contain:
- `best_model.pth` - Best performing model checkpoint
- `last_model.pth` - Latest model checkpoint
- Training logs and metrics

### Inference

Run inference on images or folders:

```bash
# Single image inference
python inference_rfdetr.py \
    --model ../../weights/rfdetr_vineyard/best_model.pth \
    --input /path/to/image.jpg \
    --output ../../data/output/rfdetr_inference

# Folder inference
python inference_rfdetr.py \
    --model ../../weights/rfdetr_vineyard/best_model.pth \
    --input /path/to/image/folder/ \
    --output ../../data/output/rfdetr_inference

# With custom confidence threshold
python inference_rfdetr.py \
    --model ../../weights/rfdetr_vineyard/best_model.pth \
    --input /path/to/images/ \
    --conf 0.6 \
    --output ../../data/output/rfdetr_inference
```

#### Inference Parameters

- `--model, -m`: Path to trained RF-DETR model (required)
- `--input, -i`: Path to input image or folder (required)
- `--output, -o`: Output directory for results (default: '../../data/output/rfdetr_inference')
- `--conf, -c`: Confidence threshold (default: 0.5)
- `--device`: Device to use - 'cuda', 'cpu', 'auto' (default: 'auto')
- `--no-viz`: Skip visualization
- `--no-json`: Skip JSON output
- `--show`: Display images interactively (single image mode only)

#### Output

The inference script creates:
```
output_directory/
├── visualizations/          # Annotated images with bounding boxes
│   ├── image1_annotated.jpg
│   └── image2_annotated.jpg
├── json/                    # Detection results in JSON format
│   ├── image1.json
│   └── image2.json
└── summary.json            # Summary of all detections
```

## Examples

### Example 1: Train a model with default settings

```bash
cd /home/cheddar/code/vineyard_detection/scripts/roboflow_rfdetr
python train_rfdetr.py
```

### Example 2: Train with larger batch size and more epochs

```bash
python train_rfdetr.py --epochs 200 --batch-size 16 --model-size x
```

### Example 3: Run inference on a single image with visualization

```bash
python inference_rfdetr.py \
    --model ../../weights/rfdetr_vineyard/best_model.pth \
    --input ../../data/test_images/vineyard_image.jpg \
    --show
```

### Example 4: Batch inference on a folder

```bash
python inference_rfdetr.py \
    --model ../../weights/rfdetr_vineyard/best_model.pth \
    --input ../../data/test_images/ \
    --conf 0.6 \
    --output ../../data/output/batch_results
```

## Model Performance

The model performance metrics will be displayed during training:
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
- Per-class Average Precision for each detection class

## Troubleshooting

### GPU Memory Issues

If you encounter GPU memory errors during training:
```bash
# Reduce batch size
python train_rfdetr.py --batch-size 2

# Or reduce image size
python train_rfdetr.py --img-size 512
```

### CUDA Not Available

If CUDA is not available or you want to use CPU:
```bash
python train_rfdetr.py --device cpu
```

Note: Training on CPU will be significantly slower.

### Import Errors

If you get import errors:
```bash
# Reinstall RF-DETR
pip uninstall rfdetr
pip install rfdetr

# Or install from source
git clone https://github.com/roboflow/rf-detr.git
cd rf-detr
pip install -e .
```

## Additional Notes

### Class Visualization Colors

In the visualization output:
- **Vineyard**: Blue boxes
- **Pole**: Green boxes
- **Trunk**: Red boxes  
- **Vine Row**: Cyan boxes

### JSON Output Format

Each detection JSON file contains:
```json
[
  {
    "class_id": 1,
    "class_name": "pole",
    "confidence": 0.95,
    "bbox": [100, 200, 150, 300]
  }
]
```
Where `bbox` is `[x1, y1, x2, y2]` in pixel coordinates.

## References

- RF-DETR Documentation: https://rfdetr.roboflow.com/latest/
- Roboflow: https://roboflow.com
- Dataset location: `code/vineyard_detection/data/datasets/datasets_coco/vineyard_segmentation_paper-2`

## License

Please refer to the main project license.
