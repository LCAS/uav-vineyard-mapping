from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image

# --- Config ---
MODEL_PATH = "models/patch_images/yolo/trained/vineyard_object_detection_obb_1/weights/best.pt"  # YOLO OBB model
IMAGE_PATH = "dataset/orthophoto/jojo/may_2025/patches/3_6.png"
OUTPUT_FOLDER = "./yolo_inference_outputs/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Load YOLO model ---
model = YOLO(MODEL_PATH)
print("YOLO OBB model loaded.")

# --- Inference ---
results = model.predict(
    source=IMAGE_PATH,
    imgsz=800,
    conf=0.25,   # confidence threshold
    iou=0.45,    # NMS IoU threshold
    save=False,
    device="cuda" if model.device.type == "cuda" else "cpu"
)

# --- Load image for visualization ---
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else []
    scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else []
    classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else []

    # For OBB, YOLO returns rotated boxes as (x_center, y_center, width, height, angle)
    obb_boxes = r.boxes.xywhr.cpu().numpy() if hasattr(r.boxes, "xywhr") else []

    for i, box in enumerate(obb_boxes):
        cx, cy, w, h, angle = box
        cls_id = int(classes[i])
        conf = float(scores[i])

        # Draw rotated rectangle
        rect = ((cx, cy), (w, h), angle)
        box_pts = cv2.boxPoints(rect).astype(int)
        color = (255, 0, 0) if cls_id == 0 else (0, 255, 0)
        cv2.polylines(image_rgb, [box_pts], isClosed=True, color=color, thickness=2)
        cv2.putText(image_rgb, f"{cls_id}:{conf:.2f}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# --- Save outputs ---
output_image_path = os.path.join(OUTPUT_FOLDER, "yolo_obb_detection.png")
Image.fromarray(image_rgb).save(output_image_path)
print(f"Inference complete. Output saved to: {output_image_path}")
