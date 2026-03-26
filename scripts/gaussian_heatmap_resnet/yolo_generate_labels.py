import os
import cv2
import numpy as np
from PIL import Image

# --- Config ---
PATCH_DIR = "dataset_split/test/images"
POST_MASK_DIR = "dataset_split/test/post_masks"
ROW_MASK_DIR = "dataset_split/test/row_masks"
YOLO_LABELS_DIR = "dataset_split/test/labels"

os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

def mask_to_obb_labels(mask_path, class_index, img_w, img_h):
    """Convert binary mask to YOLO OBB label lines."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)  # 4 corners as float

        # normalize & clip to [0, 0.999999] to avoid exact 1.0
        norm_box = []
        for (x, y) in box:
            nx = np.clip(x / img_w, 0, 0.999999)
            ny = np.clip(y / img_h, 0, 0.999999)
            norm_box.extend([nx, ny])

        label_line = f"{class_index} " + " ".join([f"{v:.6f}" for v in norm_box])
        labels.append(label_line)

    return labels

def process_dataset(patch_dir, post_mask_dir, row_mask_dir, labels_dir):
    for fname in os.listdir(patch_dir):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(patch_dir, fname)
        post_mask_path = os.path.join(post_mask_dir, fname)
        row_mask_path = os.path.join(row_mask_dir, fname)
        label_path = os.path.join(labels_dir, fname.replace(".png", ".txt"))

        # image size
        img = Image.open(img_path)
        img_w, img_h = img.size

        labels = []
        # posts = class 0
        if os.path.exists(post_mask_path):
            labels.extend(mask_to_obb_labels(post_mask_path, 0, img_w, img_h))
        # rows = class 1
        if os.path.exists(row_mask_path):
            labels.extend(mask_to_obb_labels(row_mask_path, 1, img_w, img_h))

        if labels:
            with open(label_path, "w") as f:
                f.write("\n".join(labels))
            print(f"✅ Saved {len(labels)} OBB labels → {label_path}")
        else:
            print(f"⚠️ No objects found in {fname}")

# --- Run ---
if __name__ == "__main__":
    process_dataset(PATCH_DIR, POST_MASK_DIR, ROW_MASK_DIR, YOLO_LABELS_DIR)
    print("🎯 Done! YOLO-OBB labels generated.")
