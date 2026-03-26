import os
import cv2
import shutil
import numpy as np
from PIL import Image, ImageFilter

# ========================
# CONFIGURATION
# ========================
DATASET_ROOT = "../../data/datasets/vineyard_segmentation_paper-1"
OUTPUT_ROOT = "heatmap_masks_from_yolo_labels/vineyard_segmentation_paper-1"

GAUSSIAN_RADIUS = 20    # controls gaussian spread (sigma) for bbox centers
BLUR_RADIUS = 15        # gaussian blur sigma for the final mask image
DRAW_BOXES = False      # if True, fill bbox; if False, draw gaussian at center
SAVE_OVERLAY = True     # save color overlay visualization

# class -> folder name
CLASS_NAMES = {
    0: "pole",
    1: "trunk",
    2: "vine_row"
}

# ========================
# HELPERS
# ========================
def draw_gaussian(mask, center, sigma):
    """Safely draws a gaussian (values in [0,1]) on float32 mask."""
    x, y = center
    h, w = mask.shape

    x0, x1 = int(np.floor(x - 3 * sigma)), int(np.ceil(x + 3 * sigma))
    y0, y1 = int(np.floor(y - 3 * sigma)), int(np.ceil(y + 3 * sigma))

    xmin, xmax = max(0, x0), min(w, x1)
    ymin, ymax = max(0, y0), min(h, y1)

    if xmin >= xmax or ymin >= ymax:
        return

    xs, ys = np.arange(xmin, xmax), np.arange(ymin, ymax)
    xx, yy = np.meshgrid(xs, ys)
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    patch = mask[ymin:ymax, xmin:xmax]
    if gaussian.shape != patch.shape:
        gaussian = gaussian[:patch.shape[0], :patch.shape[1]]

    mask[ymin:ymax, xmin:xmax] = np.maximum(patch, gaussian)


def parse_label_line(line):
    # <<< --- THIS IS THE CORRECTED LOGIC --- >>>
    """
    Parses YOLO label line into bbox or polygon info based on coordinate count.
    This works for any class ID.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    cls = int(float(parts[0]))
    coords = list(map(float, parts[1:]))

    # If it has 4 coordinate points, it's a bounding box.
    if len(coords) == 4:
        return {"cls": cls, "type": "bbox", "coords": coords}
    # If it has 6 or more and an even number of points, it's a polygon.
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        return {"cls": cls, "type": "poly", "coords": coords}
    # Otherwise, the line is invalid.
    else:
        return None


def ensure_dirs_for_subset(base_out, subset):
    """Creates the necessary output directory structure."""
    subset_out = os.path.join(base_out, subset)
    images_out = os.path.join(subset_out, "images")
    overlay_out = os.path.join(subset_out, "overlay")
    masks_root = os.path.join(subset_out, "masks")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(overlay_out, exist_ok=True)
    os.makedirs(masks_root, exist_ok=True)

    mask_dirs = {}
    for cls, name in CLASS_NAMES.items():
        cls_dir = os.path.join(masks_root, name)
        os.makedirs(cls_dir, exist_ok=True)
        mask_dirs[cls] = cls_dir

    return images_out, overlay_out, mask_dirs


def process_image(image_path, label_path, mask_out_paths, overlay_out_path):
    """Generates masks and a combined JET heatmap overlay."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    height, width = image.shape[:2]
    masks = {cls: np.zeros((height, width), dtype=np.float32) for cls in CLASS_NAMES}

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parsed = parse_label_line(line)
                if not parsed: continue
                
                cls = parsed["cls"]
                if cls not in masks: continue
                
                if parsed["type"] == "poly":
                    coords = np.array(parsed["coords"]).reshape(-1, 2)
                    coords[:, 0] *= width
                    coords[:, 1] *= height
                    pts = coords.astype(np.int32)
                    cv2.fillPoly(masks[cls], [pts], 1)
                
                elif parsed["type"] == "bbox":
                    x_c, y_c, w, h = parsed["coords"]
                    x_c, y_c = int(x_c * width), int(y_c * height)
                    
                    if DRAW_BOXES:
                        w, h = int(w * width), int(h * height)
                        x1, y1 = x_c - w // 2, y_c - h // 2
                        x2, y2 = x_c + w // 2, y_c + h // 2
                        cv2.rectangle(masks[cls], (x1, y1), (x2, y2), 1, -1)
                    else:
                        draw_gaussian(masks[cls], (x_c, y_c), GAUSSIAN_RADIUS)

    # Save individual class masks
    for cls, mask in masks.items():
        if np.any(mask):
            mask_img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
            if BLUR_RADIUS > 0:
                mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
            mask_img.save(mask_out_paths[cls])

    # Create and save the combined JET heatmap overlay
    if SAVE_OVERLAY:
        merged_mask = np.clip(sum(masks.values()), 0, 1)
        detection_mask = merged_mask > 0 # Mask of where any detection exists
        
        heatmap = cv2.applyColorMap((merged_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend heatmap and original image
        blended_part = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # Create a copy of the original and apply the blend only on detected areas
        output_image = image.copy()
        output_image[detection_mask] = blended_part[detection_mask]
        
        cv2.imwrite(overlay_out_path, output_image)


# ========================
# MAIN LOOP
# ========================
def main():
    for subset in ["train", "valid", "test"]:
        images_dir = os.path.join(DATASET_ROOT, subset, "images")
        labels_dir = os.path.join(DATASET_ROOT, subset, "labels")

        if not os.path.isdir(images_dir):
            print(f"[INFO] Subset not found, skipping: {subset}")
            continue

        images_out, overlay_out, mask_dirs = ensure_dirs_for_subset(OUTPUT_ROOT, subset)

        print(f"--- Processing subset: {subset} ---")
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(images_dir, fname)
            lbl_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")

            shutil.copy2(img_path, os.path.join(images_out, fname))

            mask_out_paths = {
                cls: os.path.join(mask_dirs[cls], os.path.splitext(fname)[0] + "_mask.png")
                for cls in mask_dirs
            }
            overlay_out_path = os.path.join(overlay_out, os.path.splitext(fname)[0] + "_overlay.jpg")
            
            process_image(img_path, lbl_path, mask_out_paths, overlay_out_path)

    print(f"\nAll done. Output written to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()