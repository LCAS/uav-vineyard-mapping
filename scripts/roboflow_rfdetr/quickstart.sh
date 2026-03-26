#!/bin/bash

# Quick start script for RF-DETR Vineyard Detection
# This script helps set up and run training/inference quickly

set -e  # Exit on error

echo "=================================="
echo "RF-DETR Vineyard Detection Setup"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Function to install dependencies
install_deps() {
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "Dependencies installed successfully!"
    echo ""
}

# Function to train model
train_model() {
    echo "Starting RF-DETR training..."
    echo ""
    
    # Default training parameters
    EPOCHS=${EPOCHS:-100}
    BATCH_SIZE=${BATCH_SIZE:-4}
    IMG_SIZE=${IMG_SIZE:-640}
    MODEL_SIZE=${MODEL_SIZE:-l}
    
    echo "Training parameters:"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Image size: $IMG_SIZE"
    echo "  Model size: $MODEL_SIZE"
    echo ""
    
    python3 train_rfdetr.py \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --img-size $IMG_SIZE \
        --model-size $MODEL_SIZE
}

# Function to run inference
run_inference() {
    echo "Running RF-DETR inference..."
    echo ""
    
    if [ -z "$1" ]; then
        echo "Error: Please provide input path"
        echo "Usage: ./quickstart.sh inference <input_path> [model_path] [confidence]"
        exit 1
    fi
    
    INPUT_PATH=$1
    MODEL_PATH=${2:-"../../weights/rfdetr_vineyard/best_model.pth"}
    CONF=${3:-0.5}
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Model not found at $MODEL_PATH"
        echo "Please train a model first using: ./quickstart.sh train"
        exit 1
    fi
    
    echo "Inference parameters:"
    echo "  Input: $INPUT_PATH"
    echo "  Model: $MODEL_PATH"
    echo "  Confidence: $CONF"
    echo ""
    
    python3 inference_rfdetr.py \
        --model "$MODEL_PATH" \
        --input "$INPUT_PATH" \
        --conf $CONF \
        --output "../../data/output/rfdetr_inference"
}

# Function to check dataset
check_dataset() {
    echo "Checking dataset..."
    echo ""
    
    DATASET_PATH="../../data/datasets/datasets_coco/vineyard_segmentation_paper-2"
    
    if [ ! -d "$DATASET_PATH" ]; then
        echo "Error: Dataset not found at $DATASET_PATH"
        exit 1
    fi
    
    if [ ! -f "$DATASET_PATH/train/_annotations.coco.json" ]; then
        echo "Error: Training annotations not found"
        exit 1
    fi
    
    echo "Dataset found at: $DATASET_PATH"
    echo ""
    
    # Count images
    TRAIN_IMAGES=$(find "$DATASET_PATH/train" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
    VALID_IMAGES=$(find "$DATASET_PATH/valid" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    TEST_IMAGES=$(find "$DATASET_PATH/test" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    
    echo "Dataset statistics:"
    echo "  Training images: $TRAIN_IMAGES"
    echo "  Validation images: $VALID_IMAGES"
    echo "  Test images: $TEST_IMAGES"
    echo ""
    
    # Show categories
    if command -v jq &> /dev/null; then
        echo "Classes in dataset:"
        jq '.categories[] | "\(.id): \(.name)"' "$DATASET_PATH/train/_annotations.coco.json"
        echo ""
    fi
}

# Function to show help
show_help() {
    cat << EOF
RF-DETR Vineyard Detection - Quick Start Script

Usage:
    ./quickstart.sh [command] [options]

Commands:
    install             Install required dependencies
    check              Check dataset availability
    train              Train RF-DETR model
    inference <input>  Run inference on image/folder
    help               Show this help message

Examples:
    # Install dependencies
    ./quickstart.sh install
    
    # Check dataset
    ./quickstart.sh check
    
    # Train with default settings
    ./quickstart.sh train
    
    # Train with custom parameters
    EPOCHS=200 BATCH_SIZE=8 ./quickstart.sh train
    
    # Run inference on single image
    ./quickstart.sh inference /path/to/image.jpg
    
    # Run inference with custom confidence
    ./quickstart.sh inference /path/to/images/ ../../weights/rfdetr_vineyard/best_model.pth 0.6

Environment Variables:
    EPOCHS       - Number of training epochs (default: 100)
    BATCH_SIZE   - Training batch size (default: 4)
    IMG_SIZE     - Input image size (default: 640)
    MODEL_SIZE   - Model size: s, m, l, x (default: l)

EOF
}

# Main script logic
case "$1" in
    install)
        install_deps
        ;;
    check)
        check_dataset
        ;;
    train)
        check_dataset
        train_model
        ;;
    inference)
        run_inference "${@:2}"
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "Error: Unknown command '$1'"
        echo ""
        show_help
        exit 1
        ;;
esac

echo "Done!"
