"""
Example script demonstrating how to use RF-DETR for vineyard detection

This script shows various ways to use the training and inference code programmatically.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import the modules (assuming they're properly installed)
try:
    from rfdetr import RFDETR
    from inference_rfdetr import VineyardDetector
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure rf-detr is installed: pip install rf-detr")
    sys.exit(1)


def example_1_simple_training():
    """
    Example 1: Simple training with default parameters
    """
    print("\n" + "="*80)
    print("Example 1: Simple Training")
    print("="*80)
    
    # Dataset paths
    dataset_base = "../../data/datasets/datasets_coco/vineyard_segmentation_paper-2"
    
    # Initialize model
    model = RFDETR(
        model_size='l',
        num_classes=4,
        pretrained=True
    )
    
    # Setup training
    model.setup_training(
        train_images_dir=f"{dataset_base}/train",
        train_annotations=f"{dataset_base}/train/_annotations.coco.json",
        val_images_dir=f"{dataset_base}/valid",
        val_annotations=f"{dataset_base}/valid/_annotations.coco.json",
        batch_size=4,
        num_workers=4,
        img_size=640
    )
    
    # Train
    best_model = model.train(
        epochs=10,  # Using small number for quick example
        learning_rate=1e-4,
        output_dir="../../weights/rfdetr_vineyard_example",
        device='cuda'  # or 'cpu'
    )
    
    print(f"Training complete! Best model saved at: {best_model}")


def example_2_custom_training_config():
    """
    Example 2: Training with custom configuration
    """
    print("\n" + "="*80)
    print("Example 2: Custom Training Configuration")
    print("="*80)
    
    config = {
        'model_size': 'x',  # Extra large model
        'num_classes': 4,
        'pretrained': True,
        'epochs': 100,
        'batch_size': 8,
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'img_size': 800,  # Larger images
        'patience': 20,
        'warmup_epochs': 5
    }
    
    # Initialize and train
    model = RFDETR(
        model_size=config['model_size'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained']
    )
    
    # ... rest of training setup


def example_3_single_image_inference():
    """
    Example 3: Run inference on a single image
    """
    print("\n" + "="*80)
    print("Example 3: Single Image Inference")
    print("="*80)
    
    # Path to trained model and test image
    model_path = "../../weights/rfdetr_vineyard/best_model.pth"
    image_path = "../../data/test_images/vineyard_sample.jpg"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first!")
        return
    
    # Initialize detector
    detector = VineyardDetector(
        model_path=model_path,
        device='auto',
        conf_threshold=0.5
    )
    
    # Run inference
    detections = detector.predict(image_path)
    
    # Print results
    print(f"\nFound {len(detections)} objects:")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['class_name']}: {det['confidence']:.3f} at {det['bbox']}")
    
    # Visualize
    output_path = "../../data/output/example_inference.jpg"
    detector.visualize(image_path, detections, save_path=output_path)
    print(f"\nVisualization saved to: {output_path}")


def example_4_batch_inference():
    """
    Example 4: Run inference on multiple images
    """
    print("\n" + "="*80)
    print("Example 4: Batch Inference")
    print("="*80)
    
    model_path = "../../weights/rfdetr_vineyard/best_model.pth"
    images_folder = "../../data/test_images/"
    output_dir = "../../data/output/batch_inference"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Initialize detector
    detector = VineyardDetector(model_path=model_path)
    
    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(images_folder).glob(ext))
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    all_results = []
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        
        # Run inference
        detections = detector.predict(str(img_path))
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{img_path.stem}_annotated.jpg")
        detector.visualize(str(img_path), detections, save_path=output_path)
        
        # Collect results
        result = {
            'image': img_path.name,
            'detections': len(detections),
            'poles': sum(1 for d in detections if d['class_name'] == 'pole'),
            'trunks': sum(1 for d in detections if d['class_name'] == 'trunk'),
            'vine_rows': sum(1 for d in detections if d['class_name'] == 'vine_row')
        }
        all_results.append(result)
        print(f"  Found: {result['poles']} poles, {result['trunks']} trunks, "
              f"{result['vine_rows']} vine rows")
    
    # Print summary
    print("\n" + "-"*80)
    print("Summary:")
    total_poles = sum(r['poles'] for r in all_results)
    total_trunks = sum(r['trunks'] for r in all_results)
    total_vine_rows = sum(r['vine_rows'] for r in all_results)
    
    print(f"Total images: {len(all_results)}")
    print(f"Total poles: {total_poles}")
    print(f"Total trunks: {total_trunks}")
    print(f"Total vine rows: {total_vine_rows}")


def example_5_evaluate_model():
    """
    Example 5: Evaluate model on test set
    """
    print("\n" + "="*80)
    print("Example 5: Model Evaluation")
    print("="*80)
    
    model_path = "../../weights/rfdetr_vineyard/best_model.pth"
    test_dir = "../../data/datasets/datasets_coco/vineyard_segmentation_paper-2/test"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Load model
    model = RFDETR.load(model_path)
    
    # Evaluate
    results = model.evaluate(
        images_dir=test_dir,
        annotations=f"{test_dir}/_annotations.coco.json",
        device='auto'
    )
    
    # Print metrics
    print("\nTest Set Results:")
    print(f"mAP@0.5: {results.get('mAP@0.5', 'N/A'):.4f}")
    print(f"mAP@0.5:0.95: {results.get('mAP@0.5:0.95', 'N/A'):.4f}")
    
    # Per-class results
    if 'per_class' in results:
        print("\nPer-class Average Precision:")
        class_names = ['vineyard', 'pole', 'trunk', 'vine_row']
        for class_name, ap in zip(class_names, results['per_class']):
            print(f"  {class_name}: {ap:.4f}")


def example_6_inference_with_filtering():
    """
    Example 6: Inference with custom filtering and post-processing
    """
    print("\n" + "="*80)
    print("Example 6: Inference with Filtering")
    print("="*80)
    
    model_path = "../../weights/rfdetr_vineyard/best_model.pth"
    image_path = "../../data/test_images/vineyard_sample.jpg"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Initialize detector
    detector = VineyardDetector(
        model_path=model_path,
        conf_threshold=0.3  # Lower threshold to get more detections
    )
    
    # Run inference
    detections = detector.predict(image_path)
    
    # Filter only poles with high confidence
    poles = [d for d in detections 
             if d['class_name'] == 'pole' and d['confidence'] > 0.7]
    
    print(f"Total detections: {len(detections)}")
    print(f"High-confidence poles: {len(poles)}")
    
    # Sort by confidence
    poles_sorted = sorted(poles, key=lambda x: x['confidence'], reverse=True)
    
    print("\nTop 5 poles by confidence:")
    for i, pole in enumerate(poles_sorted[:5], 1):
        print(f"  {i}. Confidence: {pole['confidence']:.3f}, "
              f"Location: ({pole['bbox'][0]:.0f}, {pole['bbox'][1]:.0f})")


def main():
    """
    Main function to run examples
    """
    print("\n" + "="*80)
    print("RF-DETR Vineyard Detection - Usage Examples")
    print("="*80)
    
    examples = {
        '1': ('Simple Training', example_1_simple_training),
        '2': ('Custom Training Config', example_2_custom_training_config),
        '3': ('Single Image Inference', example_3_single_image_inference),
        '4': ('Batch Inference', example_4_batch_inference),
        '5': ('Model Evaluation', example_5_evaluate_model),
        '6': ('Inference with Filtering', example_6_inference_with_filtering),
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("\nTo run an example:")
    print("  python examples.py <example_number>")
    print("\nExample:")
    print("  python examples.py 3")
    
    # Run example if specified
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"\nRunning: {name}")
            func()
        else:
            print(f"\nError: Invalid example number '{example_num}'")
    else:
        print("\nNo example specified. Use 'python examples.py <number>' to run.")


if __name__ == "__main__":
    main()
