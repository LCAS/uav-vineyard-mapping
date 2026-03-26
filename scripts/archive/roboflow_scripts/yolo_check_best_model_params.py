from ultralytics import YOLO

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Replace this with the actual path to your best.pt file
model_path = '../../data/datasets/trained/vineyard_segmentation-22/train5/weights/best.pt' 

# Replace this with the path to your data.yaml file
data_yaml_path = '../../data/datasets/vineyard_segmentation-22/data.yaml'

# Load the model
model = YOLO(model_path)

# ---------------------------------------------------------
# 1. Evaluate on the TEST set (To compare with Roboflow)
# ---------------------------------------------------------
print("\n" + "="*40)
print("🧪 1. RUNNING VALIDATION ON 'TEST' SPLIT")
print("   (This usually matches Roboflow metrics)")
print("="*40)

try:
    test_results = model.val(
        data=data_yaml_path,
        split='test',       # Forces use of the test set images
        project='runs/val', # Where to save the output images
        name='test_evaluation'
    )
    
    print("\n✅ TEST SET METRICS:")
    print(f"   Box mAP@50:    {test_results.box.map50:.1%}")
    print(f"   Box Precision: {test_results.box.mp:.1%}")
    print(f"   Box Recall:    {test_results.box.mr:.1%}")
    print("-" * 20)
    print(f"   Mask mAP@50:   {test_results.seg.map50:.1%}")
    print(f"   Mask Precision:{test_results.seg.mp:.1%}")
    print(f"   Mask Recall:   {test_results.seg.mr:.1%}")

except Exception as e:
    print(f"\n❌ Could not run on Test set. Error: {e}")
    print("   (Check if your data.yaml actually has a 'test:' path defined)")


# ---------------------------------------------------------
# 2. Evaluate on the VAL set (To compare with CSV Logs)
# ---------------------------------------------------------
print("\n" + "="*40)
print("📉 2. RUNNING VALIDATION ON 'VAL' SPLIT")
print("   (This usually matches your CSV logs/Epoch 27)")
print("="*40)

val_results = model.val(
    data=data_yaml_path,
    split='val',
    project='runs/val',
    name='val_evaluation'
)

print("\n✅ VAL SET METRICS:")
print(f"   Box mAP@50:    {val_results.box.map50:.1%}")
print(f"   Box Precision: {val_results.box.mp:.1%}")
print(f"   Box Recall:    {val_results.box.mr:.1%}")
print("-" * 20)
print(f"   Mask mAP@50:   {val_results.seg.map50:.1%}")
print(f"   Mask Precision:{val_results.seg.mp:.1%}")
print(f"   Mask Recall:   {val_results.seg.mr:.1%}")