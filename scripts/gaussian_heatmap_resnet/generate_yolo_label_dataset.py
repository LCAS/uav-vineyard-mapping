import os
import shutil
from tqdm import tqdm
import cv2
import random
import numpy as np
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

# ==========================================================
# CONFIGURATION
# ==========================================================
IMAGE_DIRS = [
    "../../images/riseholme/august_2024/39_feet",
    "../../images/riseholme/august_2024/65_feet",
    "../../images/riseholme/august_2024/100_feet",
    "../../images/riseholme/march_2025/39_feet",
    "../../images/riseholme/march_2025/65_feet",
    "../../images/riseholme/march_2025/100_feet"
]

LABEL_DIRS = {
    "train": "../../data/datasets/vineyard_segmentation-18/train/labels",
    "valid": "../../data/datasets/vineyard_segmentation-18/valid/labels",
    "test":  "../../data/datasets/vineyard_segmentation-18/test/labels"
}

OUTPUT_BASE = "dataset_yolov11_labels"
PATCH_SIZES = [(960, 1280), (480, 640)]  # (height, width)
OVERLAP = 0.2  # 20% overlap for patches

# ==========================================================
# UTILITIES
# ==========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_all_images():
    images = []
    for img_dir in IMAGE_DIRS:
        for f in os.listdir(img_dir):
            if f.lower().endswith(".jpg"):
                images.append(os.path.join(img_dir, f))
    return sorted(images)

def match_label(image_name, label_files):
    base_no_ext = os.path.splitext(image_name)[0]
    for lbl in label_files:
        if base_no_ext in lbl:
            return lbl
    return None

def get_class_color(cls_id):
    random.seed(cls_id)
    return tuple(int(x) for x in np.random.randint(50, 255, 3))

# ==========================================================
# ANNOTATION
# ==========================================================
def draw_yolo_polygons(image_path, output_path, label_lines=None, crop_coords=None):
    """Draw YOLO polygon labels; label_lines can be passed instead of reading from a file."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Could not read {image_path}")
        return

    if crop_coords:
        x_off, y_off, patch_h, patch_w = crop_coords
        image = image[y_off:y_off+patch_h, x_off:x_off+patch_w]

    h, w = image.shape[:2]

    if label_lines is None:
        return

    for line in label_lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        pts = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            pts.append([x, y])

        if len(pts) < 3:
            continue

        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        color = get_class_color(cls)

        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

        text_pos = tuple(pts[0][0])
        cv2.putText(image, f"Class {cls}", (text_pos[0], max(20, text_pos[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, image)

# ==========================================================
# PATCHING
# ==========================================================
def tile_image(img, patch_size, overlap):
    h, w = img.shape[:2]
    ph, pw = patch_size
    stride_y = int(ph * (1 - overlap))
    stride_x = int(pw * (1 - overlap))
    patches = []

    y_positions = list(range(0, h - ph + 1, stride_y))
    x_positions = list(range(0, w - pw + 1, stride_x))
    if y_positions[-1] != h - ph:
        y_positions.append(h - ph)
    if x_positions[-1] != w - pw:
        x_positions.append(w - pw)

    for y in y_positions:
        for x in x_positions:
            patch = img[y:y+ph, x:x+pw]
            patches.append((x, y, patch))
    return patches

# ==========================================================
# PATCH LABEL CROPPING
# ==========================================================
def crop_yolo_labels_for_patch(label_path, crop_coords, full_img_shape):
    """Return YOLO label lines for the patch, handling invalid polygons."""
    x_off, y_off, ph, pw = crop_coords
    full_h, full_w = full_img_shape[:2]
    patch_lines = []

    patch_box = box(0, 0, pw, ph)

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        abs_pts = [(coords[i] * full_w - x_off, coords[i+1] * full_h - y_off)
                   for i in range(0, len(coords), 2)]
        poly = Polygon(abs_pts)

        # Fix invalid polygons
        if not poly.is_valid:
            poly = make_valid(poly)
            # make_valid can return MultiPolygon, handle that
            if poly.is_empty:
                continue
            if poly.geom_type == "MultiPolygon":
                # We'll just take the largest polygon by area
                poly = max(poly.geoms, key=lambda p: p.area)

        # Intersect with patch
        poly_clipped = poly.intersection(patch_box)
        if poly_clipped.is_empty or not poly_clipped.is_valid:
            continue

        if hasattr(poly_clipped, "exterior"):
            pts = list(poly_clipped.exterior.coords)
            if len(pts) < 3:
                continue
            norm_pts = []
            for x, y in pts[:-1]:
                x = min(max(x, 0), pw - 1)
                y = min(max(y, 0), ph - 1)
                norm_pts.extend([x / pw, y / ph])
            patch_lines.append(f"{cls} " + " ".join([f"{p:.6f}" for p in norm_pts]))

    return patch_lines

# ==========================================================
# MAIN
# ==========================================================
def main():
    images = get_all_images()
    print(f"Found {len(images)} total images across all folders.")

    for subset, label_dir in LABEL_DIRS.items():
        print(f"\nProcessing {subset} subset...")
        label_files = os.listdir(label_dir)

        # --- Full resolution folders ---
        full_res_dir = os.path.join(OUTPUT_BASE, "full_res", subset)
        full_img_dir = os.path.join(full_res_dir, "images")
        full_lbl_dir = os.path.join(full_res_dir, "labels")
        full_ann_dir = os.path.join(full_res_dir, "annotated_images")
        for d in [full_img_dir, full_lbl_dir, full_ann_dir]:
            ensure_dir(d)

        matched, unmatched = 0, 0

        for img_path in tqdm(images, desc=f"Matching {subset}"):
            img_name = os.path.basename(img_path)
            label_name = match_label(img_name, label_files)
            if not label_name:
                unmatched += 1
                continue
            matched += 1

            # --- Copy full resolution ---
            dst_img = os.path.join(full_img_dir, img_name)
            shutil.copy(img_path, dst_img)
            dst_label = os.path.join(full_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
            shutil.copy(os.path.join(label_dir, label_name), dst_label)
            dst_annotated = os.path.join(full_ann_dir, img_name)
            with open(dst_label, "r") as f:
                full_label_lines = f.readlines()
            draw_yolo_polygons(dst_img, dst_annotated, label_lines=full_label_lines)

            # --- Patches ---
            image = cv2.imread(img_path)
            if image is None:
                continue
            for ph, pw in PATCH_SIZES:
                patches = tile_image(image, (ph, pw), OVERLAP)
                patch_base_dir = os.path.join(OUTPUT_BASE, f"patches_{ph}x{pw}", subset)
                patch_img_dir = os.path.join(patch_base_dir, "images")
                patch_lbl_dir = os.path.join(patch_base_dir, "labels")
                patch_ann_dir = os.path.join(patch_base_dir, "annotated_images")
                for d in [patch_img_dir, patch_lbl_dir, patch_ann_dir]:
                    ensure_dir(d)

                for x_off, y_off, patch in patches:
                    patch_name = f"{os.path.splitext(img_name)[0]}_patch_{x_off}_{y_off}.jpg"
                    patch_img_path = os.path.join(patch_img_dir, patch_name)
                    cv2.imwrite(patch_img_path, patch)

                    patch_label_path = os.path.join(patch_lbl_dir, os.path.splitext(patch_name)[0] + ".txt")
                    patch_lines = crop_yolo_labels_for_patch(
                        os.path.join(label_dir, label_name),
                        (x_off, y_off, ph, pw),
                        image.shape
                    )

                    with open(patch_label_path, "w") as pf:
                        for pl in patch_lines:
                            pf.write(pl + "\n")

                    patch_annotated_path = os.path.join(patch_ann_dir, patch_name)
                    draw_yolo_polygons(patch_img_path, patch_annotated_path, label_lines=patch_lines)

        print(f"✅ {matched} matched labels copied for {subset}.")
        if unmatched > 0:
            print(f"⚠️ {unmatched} images had no matching label in {subset}.")

if __name__ == "__main__":
    main()
