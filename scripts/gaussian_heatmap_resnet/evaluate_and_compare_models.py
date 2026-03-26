#!/usr/bin/env python3
"""
Evaluate and Compare ResNet Training Results
Analyzes metrics from all trained models (ResNet18, ResNet50, ResNet101) 
and determines the best performing model.
"""

import os
import csv
import json
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None

# Configuration
RESULTS_DIR = "results_resnet/yolo_masks/vineyard_segmentation_paper_1"

def load_training_config(model_dir):
    """Load training configuration from YAML file"""
    yaml_files = list(Path(model_dir).glob("*.yaml"))
    if yaml_files and yaml:
        with open(yaml_files[0], 'r') as f:
            try:
                # Use unsafe load to handle Python-specific types
                return yaml.unsafe_load(f)
            except:
                return None
    return None

def load_metrics(model_dir):
    """Load metrics from CSV file"""
    metrics_path = Path(model_dir) / "metrics.csv"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    return None

def get_best_metrics(metrics_list):
    """Extract best metrics from training history"""
    if metrics_list is None or len(metrics_list) == 0:
        return None
    
    val_losses = []
    train_losses = []
    val_accs = []
    val_ious = []
    per_class_ious = {}
    
    for i, row in enumerate(metrics_list):
        try:
            if 'val_loss' in row:
                val_losses.append((float(row['val_loss']), i))
            if 'train_loss' in row:
                train_losses.append((float(row['train_loss']), i))
            if 'pixel_acc' in row:
                val_accs.append((float(row['pixel_acc']), i))
            if 'mIoU' in row:
                val_ious.append((float(row['mIoU']), i))
            
            # Extract per-class IoU
            for key, value in row.items():
                if key.startswith('IoU_'):
                    class_name = key.replace('IoU_', '')
                    if class_name not in per_class_ious:
                        per_class_ious[class_name] = []
                    try:
                        per_class_ious[class_name].append((float(value), i))
                    except (ValueError, TypeError):
                        continue
        except (ValueError, TypeError):
            continue
    
    best_metrics = {}
    
    if val_losses:
        best_val_loss, best_epoch = min(val_losses, key=lambda x: x[0])
        best_metrics['best_val_loss'] = best_val_loss
        best_metrics['best_epoch'] = best_epoch + 1
        best_metrics['final_val_loss'] = val_losses[-1][0]
    
    if train_losses:
        best_metrics['best_train_loss'] = min(train_losses, key=lambda x: x[0])[0]
        best_metrics['final_train_loss'] = train_losses[-1][0]
    
    best_metrics['total_epochs'] = len(metrics_list)
    
    if val_accs:
        best_metrics['best_pixel_acc'] = max(val_accs, key=lambda x: x[0])[0]
        best_metrics['final_pixel_acc'] = val_accs[-1][0]
    
    if val_ious:
        best_metrics['best_miou'] = max(val_ious, key=lambda x: x[0])[0]
        best_metrics['final_miou'] = val_ious[-1][0]
    
    # Calculate best per-class IoU
    best_metrics['per_class_iou'] = {}
    for class_name, iou_list in per_class_ious.items():
        if iou_list:
            best_iou, best_epoch_class = max(iou_list, key=lambda x: x[0])
            best_metrics['per_class_iou'][class_name] = {
                'best': best_iou,
                'final': iou_list[-1][0],
                'best_epoch': best_epoch_class + 1
            }
    
    return best_metrics if best_metrics else None

def analyze_results():
    """Analyze all training results and generate comparison report"""
    
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ Results directory not found: {RESULTS_DIR}")
        return
    
    results = []
    
    # Scan all training directories
    for model_dir in sorted(os.listdir(RESULTS_DIR)):
        model_path = os.path.join(RESULTS_DIR, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        # Extract model info from directory name (e.g., train_resnet18_20260130_101212)
        parts = model_dir.split('_')
        if len(parts) < 4:
            continue
        
        # parts[0] = 'train', parts[1] = 'resnet18', parts[2] = date, parts[3] = time
        model_type = parts[1]  # e.g., "resnet18", "resnet50", "resnet101"
        timestamp = f"{parts[2]}_{parts[3]}"
        
        # Load configuration and metrics
        config = load_training_config(model_path)
        metrics = load_metrics(model_path)
        best_metrics = get_best_metrics(metrics)
        
        if config and best_metrics:
            result = {
                'model_dir': model_dir,
                'model_type': model_type,
                'timestamp': timestamp,
                'image_size': f"{config['image_size'][0]}x{config['image_size'][1]}",
                'batch_size': config['batch_size'],
                'device': config.get('device', 'unknown'),
                'learning_rate': config['learning_rate'],
                'best_val_loss': best_metrics['best_val_loss'],
                'final_val_loss': best_metrics['final_val_loss'],
                'best_miou': best_metrics.get('best_miou', None),
                'final_miou': best_metrics.get('final_miou', None),
                'best_pixel_acc': best_metrics.get('best_pixel_acc', None),
                'best_epoch': best_metrics['best_epoch'],
                'total_epochs': best_metrics['total_epochs'],
                'per_class_iou': best_metrics.get('per_class_iou', {}),
            }
            
            results.append(result)
    
    if not results:
        print("❌ No valid training results found")
        return
    
    # Print summary
    print("\n" + "="*100)
    print("MODEL COMPARISON REPORT")
    print("="*100)
    
    # Group by model type
    model_types = sorted(set(r['model_type'] for r in results))
    for model_type in model_types:
        print(f"\n{'='*100}")
        print(f"📊 {model_type.upper()}")
        print(f"{'='*100}")
        
        subset = [r for r in results if r['model_type'] == model_type]
        subset.sort(key=lambda x: x['best_val_loss'])
        
        # Display detailed comparison
        print(f"{'Model Dir':<40} {'Size':<12} {'BS':<3} {'Device':<6} {'Best Val Loss':<15} {'Epoch':<7}")
        print("-" * 100)
        for r in subset:
            epoch_str = f"{r['best_epoch']}/{r['total_epochs']}"
            print(f"{r['model_dir']:<40} {r['image_size']:<12} {r['batch_size']:<3} {r['device']:<6} {r['best_val_loss']:<15.6f} {epoch_str:<7}")
    
    # Overall best model
    print(f"\n{'='*100}")
    print("🏆 BEST MODEL BY METRIC")
    print(f"{'='*100}")
    
    best_model = min(results, key=lambda x: x['best_val_loss'])
    print(f"\n✅ Best by Validation Loss: {best_model['model_type'].upper()}")
    print(f"   Directory: {best_model['model_dir']}")
    print(f"   Best Val Loss: {best_model['best_val_loss']:.6f}")
    print(f"   Best mIoU: {best_model.get('best_miou', 'N/A')}")
    print(f"   Achieved at Epoch: {best_model['best_epoch']} / {best_model['total_epochs']}")
    print(f"   Configuration: {best_model['image_size']}, Batch={best_model['batch_size']}, Device={best_model['device']}")
    
    if best_model.get('per_class_iou'):
        print(f"\n   Per-Class IoU (excluding background):")
        for class_name in sorted(best_model['per_class_iou'].keys()):
            if class_name != 'background':
                iou = best_model['per_class_iou'][class_name]['best']
                print(f"      {class_name:<15}: {iou:.6f}")
    
    # Summary statistics by model type
    print(f"\n{'='*100}")
    print("📈 SUMMARY STATISTICS BY MODEL TYPE")
    print(f"{'='*100}\n")
    
    for model_type in model_types:
        subset = [r for r in results if r['model_type'] == model_type]
        val_losses = [r['best_val_loss'] for r in subset]
        mious = [r.get('best_miou', 0) for r in subset if r.get('best_miou')]
        
        print(f"{model_type.upper()}:")
        print(f"  Best Val Loss:   min={min(val_losses):.6f}, avg={sum(val_losses)/len(val_losses):.6f}, max={max(val_losses):.6f}")
        if mious:
            print(f"  Best mIoU:       min={min(mious):.6f}, avg={sum(mious)/len(mious):.6f}, max={max(mious):.6f}")
        print()
    
    # Per-class performance comparison
    print(f"{'='*100}")
    print("📊 PER-CLASS IoU COMPARISON (Excluding Background)")
    print(f"{'='*100}\n")
    
    # Collect all class names
    all_classes = set()
    for r in results:
        for class_name in r.get('per_class_iou', {}).keys():
            if class_name != 'background':
                all_classes.add(class_name)
    
    for class_name in sorted(all_classes):
        print(f"\n{class_name.upper()}:")
        print(f"{'Model Type':<15} {'Best IoU':<12} {'Final IoU':<12} {'Improvement':<12}")
        print("-" * 50)
        
        class_results = []
        for r in results:
            if class_name in r.get('per_class_iou', {}):
                best_iou = r['per_class_iou'][class_name]['best']
                final_iou = r['per_class_iou'][class_name]['final']
                improvement = final_iou - best_iou
                class_results.append({
                    'model_type': r['model_type'],
                    'best_iou': best_iou,
                    'final_iou': final_iou,
                    'improvement': improvement
                })
        
        for cr in sorted(class_results, key=lambda x: x['best_iou'], reverse=True):
            print(f"{cr['model_type']:<15} {cr['best_iou']:<12.6f} {cr['final_iou']:<12.6f} {cr['improvement']:<12.6f}")
    
    # Non-background average comparison
    print(f"\n{'='*100}")
    print("🎯 NON-BACKGROUND AVERAGE IoU (Pole, Trunk, Vine_Row)")
    print(f"{'='*100}\n")
    
    model_avg_iou = {}
    for r in results:
        key = f"{r['model_type']} ({r['image_size']} bs{r['batch_size']})"
        non_bg_ious = []
        for class_name, iou_data in r.get('per_class_iou', {}).items():
            if class_name != 'background':
                non_bg_ious.append(iou_data['best'])
        
        if non_bg_ious:
            avg_iou = sum(non_bg_ious) / len(non_bg_ious)
            if r['model_type'] not in model_avg_iou:
                model_avg_iou[r['model_type']] = []
            model_avg_iou[r['model_type']].append({
                'config': key,
                'avg_iou': avg_iou,
                'dir': r['model_dir']
            })
    
    print(f"{'Model Config':<50} {'Non-BG Avg IoU':<15} {'Directory':<40}\n")
    all_configs = []
    for configs in model_avg_iou.values():
        all_configs.extend(configs)
    
    for config in sorted(all_configs, key=lambda x: x['avg_iou'], reverse=True):
        print(f"{config['config']:<50} {config['avg_iou']:<15.6f} {config['dir']:<40}")
    
    # Save detailed report to CSV
    output_file = os.path.join(RESULTS_DIR, "model_comparison_report.csv")
    with open(output_file, 'w', newline='') as f:
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    print(f"✅ Detailed report saved to: {output_file}")
    
    # Save JSON summary
    json_file = os.path.join(RESULTS_DIR, "model_comparison_summary.json")
    summary_data = {
        'best_model': {
            'model_dir': best_model['model_dir'],
            'model_type': best_model['model_type'],
            'best_val_loss': best_model['best_val_loss'],
            'best_epoch': best_model['best_epoch'],
            'image_size': best_model['image_size'],
            'batch_size': best_model['batch_size'],
            'device': best_model['device'],
        },
        'all_models': results
    }
    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"✅ JSON summary saved to: {json_file}")

if __name__ == "__main__":
    analyze_results()
